"""
The multi-specimen pipeline shared by apps/app_multi_specimen.py (Gradio) and
apps/mcp/server.py (MCP server for agents such as nanobot):

    detector (RT-DETR, see training/detector/, or YOLO at a chosen resolution / auto mode)
      -> CPM keypoints kp4-kp8 for every box (one batch)
      -> per box: square crop, rotated so kp4 -> kp8 points down when both keypoints are
         confident (infer_unlabeled.py's align_crop), otherwise the same crop unrotated
      -> classifier from training/classification/finetune.py (one batch, its val transform)

Pipeline.analyze() returns the raw results; Pipeline.run() also draws the overlay and builds
the tables the app shows.  See apps/app_multi_specimen.py for the caveats.
Model files come from beetleid/foundation/paths.py (BeetleID/models/); older locations are kept as fallbacks.
"""

import colorsys
import os
import re
import sys

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import RTDETR, YOLO

from . import paths
from .infer_unlabeled import (KP_NAMES, TARGET_SIZE, align_crop, gaussian, heatmap2coord,
                                       load_model, preprocess, select_device)

# First path that exists wins: models/ first, then the old HPC layout (relative to the
# working directory), so older checkouts still run.
YOLO_WEIGHTS = [paths.DETECTOR_YOLO, "yolo_specimen_yolo11n_best.pt"]
RTDETR_WEIGHTS = [paths.DETECTOR_RTDETR]                 # round 2, see training/detector/README.md
CPM_CHECKPOINTS = [paths.KEYPOINTS_CPM,                  # best kp4/kp8 (epoch 240)
                   "ckpt/KP-Prediction-Heatmap-5+1Points-CPM-V2-FrontalProfile"
                   "-LR-Aug-Scheduler-MSE/epoch-0240.pth", "pose_ckpt/epoch-0240.pth"]
CLASSIFIERS = [paths.CLASSIFIER]
CLASS_MAPPINGS = [paths.CLASS_MAPPING, "data/mapping.txt"]
MIN_SCORE = 0.3   # same threshold as infer_unlabeled.py --align-min-score
ALIGN_PAIR = (KP_NAMES.index("kp4"), KP_NAMES.index("kp8"))   # infer_unlabeled --align-pair 4,8

# Detector choice.  RT-DETR runs at its training size; the YOLO entries are YOLO imgsz
# values, or the auto mode (see Pipeline.detect).
RTDETR_IMGSZ = 960
DETECT_RTDETR = f"RT-DETR ({RTDETR_IMGSZ})"
MAX_AREA = 0.5       # RT-DETR: drop boxes over this share of the photo when smaller ones exist
AUTO_LOW, AUTO_HIGH = 640, 1280
DETECT_AUTO = f"YOLO auto ({AUTO_LOW} + {AUTO_HIGH})"
YOLO_SIZES = ["640", "960", "1280", "1600", "1920"]
DETECT_MODES = [DETECT_RTDETR, DETECT_AUTO] + [f"YOLO {v}" for v in YOLO_SIZES]
SMALL_FRAC = 0.20    # auto: a high-res box is "small" if its long side is <= this share of the photo's
INSIDE_FRAC = 0.5    # auto: drop a high-res box if this share of it lies inside a kept box


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside_frac(a, b):
    """Share of box a's area that lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def first_existing(paths, what):
    path = next((p for p in paths if p and os.path.exists(p)), None)
    if path is None:
        raise SystemExit(f"[error] no {what} found at any of {paths}")
    return path


def palette(n):
    """n well-separated RGB colours."""
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.85, 1.0))
            for i in range(n)]


def load_class_names(path, n):
    """Class id -> name from data_splitter.py's mapping.txt ('Class: X, Class id: N, ...')."""
    names = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            m = re.match(r"Class:\s*(.+?),\s*Class id:\s*(\d+)", line.strip())
            if m:
                names[int(m.group(2))] = m.group(1)
    return [names.get(i, f"class {i}") for i in range(n)]


def load_classifier(path, device):
    """
    Rebuild a classification/finetune.py model from its checkpoint.  The checkpoint keeps
    finetune.py's args, so the architecture, class count and input size come from there:
    a resnet* model is a timm CNN, anything else a models_vit ViT, as in finetune.py.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    if a.model.startswith("resnet"):
        model = timm.create_model(a.model, pretrained=False, num_classes=a.nb_classes)
    else:
        sys.path.insert(0, paths.CLASSIFICATION_CODE)
        import models_vit
        model = models_vit.__dict__[a.model](num_classes=a.nb_classes, drop_path_rate=a.drop_path,
                                             global_pool=a.global_pool)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), a


class Pipeline:
    def __init__(self, yolo_weights, rtdetr_weights, cpm_checkpoint, classifier, class_mapping,
                 device, align_size, align_fill):
        self.device = select_device(device)
        print(f"Loading YOLO {yolo_weights}, RT-DETR {rtdetr_weights}, CPM {cpm_checkpoint}, "
              f"classifier {classifier} on {self.device} ...")
        self.yolo = YOLO(yolo_weights)
        self.rtdetr = RTDETR(rtdetr_weights) if rtdetr_weights else None
        self.cpm = load_model(cpm_checkpoint, self.device)
        self.to_tensor = transforms.ToTensor()
        cmap = gaussian(TARGET_SIZE // 2, TARGET_SIZE // 2, TARGET_SIZE, TARGET_SIZE)
        self.centermap = torch.from_numpy(cmap)[None, None].to(self.device)

        self.classifier, cargs = load_classifier(classifier, self.device)
        self.classifier_name = os.path.relpath(classifier)
        self.class_names = load_class_names(class_mapping, cargs.nb_classes)
        # finetune.py's transform_val, so the crops are seen exactly as during evaluation
        self.cls_transform = transforms.Compose([
            transforms.Resize(cargs.input_size, interpolation=3),  # 3 is bicubic
            transforms.CenterCrop(cargs.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.align_size, self.align_fill = align_size, align_fill

    def _predict(self, image_rgb, conf, iou, imgsz):
        yolo_device = 0 if self.device.type == "cuda" else "cpu"
        # Ultralytics treats a numpy array as BGR.
        res = self.yolo.predict(image_rgb[:, :, ::-1], conf=conf, iou=iou, imgsz=imgsz,
                                max_det=100, device=yolo_device, verbose=False)[0]
        return sorted(zip(res.boxes.conf.tolist(), res.boxes.xyxy.tolist()), reverse=True)

    def _predict_rtdetr(self, image_rgb, conf, iou):
        """
        RT-DETR boxes, best first.  RT-DETR has no NMS, so two clean-ups are applied here:
        a box overlapping a better one by IoU >= iou is a duplicate, and a box covering more
        than MAX_AREA of the photo while smaller boxes exist is a whole-tray box (learned from
        the single-specimen training photos).  Boxes lying inside a larger box are kept: on a
        tray that is often a real small specimen next to or under a large one.
        """
        device = 0 if self.device.type == "cuda" else "cpu"
        res = self.rtdetr.predict(image_rgb[:, :, ::-1], conf=conf, imgsz=RTDETR_IMGSZ,
                                  max_det=300, device=device, verbose=False)[0]
        dets = sorted(zip(res.boxes.conf.tolist(), res.boxes.xyxy.tolist()), reverse=True)
        area = image_rgb.shape[0] * image_rgb.shape[1]
        big = [(b[2] - b[0]) * (b[3] - b[1]) > MAX_AREA * area for _, b in dets]
        if not all(big):
            dets = [d for d, g in zip(dets, big) if not g]
        kept = []
        for s, b in dets:
            if all(box_iou(b, k) < iou for _, k in kept):
                kept.append((s, b))
        return kept

    def detect(self, image_rgb, conf, iou, mode=DETECT_RTDETR):
        """
        All specimen boxes, best first.  DETECT_RTDETR runs the RT-DETR detector (see
        _predict_rtdetr).  "YOLO <size>" runs YOLO once at that resolution.

        The detector was trained at 640, where a specimen filling the frame gets one box but
        small specimens on a tray shrink to a few pixels and are missed; at 1280+ those are
        found, but a large specimen looks bigger than anything in training and is split into
        part-sized boxes.  DETECT_AUTO takes the best of both: every 640 box, plus each 1280
        box that is small (<= SMALL_FRAC of the photo's long side) and neither overlaps
        (IoU >= iou) nor mostly lies inside (>= INSIDE_FRAC of its area) a box already kept.
        Returns (dets, n_added_from_high_res).
        """
        if mode == DETECT_RTDETR:
            if self.rtdetr is None:
                raise ValueError("No RT-DETR weights loaded; pick a YOLO mode or pass --rtdetr-weights.")
            return self._predict_rtdetr(image_rgb, conf, iou), 0
        if mode != DETECT_AUTO:
            return [(s, b) for s, b in self._predict(image_rgb, conf, iou, int(mode.split()[-1]))], 0

        kept = [(s, b) for s, b in self._predict(image_rgb, conf, iou, AUTO_LOW)]
        long_side = max(image_rgb.shape[:2])
        added = 0
        for s, b in self._predict(image_rgb, conf, iou, AUTO_HIGH):
            if max(b[2] - b[0], b[3] - b[1]) > SMALL_FRAC * long_side:
                continue   # large boxes come from the low-resolution pass
            if any(box_iou(b, k) >= iou or inside_frac(b, k) >= INSIDE_FRAC for _, k in kept):
                continue   # already covered: a duplicate, or a fragment of a kept specimen
            kept.append((s, b))
            added += 1
        return sorted(kept, key=lambda d: -d[0]), added

    @torch.no_grad()
    def keypoints(self, image_rgb, boxes):
        """CPM on every box in one batch.  Returns [(coords[5,2], scores[5])] in image px."""
        if not boxes:
            return []
        pil = Image.fromarray(image_rgb)
        canvases, transforms_ = [], []
        for box in boxes:
            canvas, scale, ox, oy = preprocess(pil, box)
            canvases.append(self.to_tensor(Image.fromarray(canvas)))
            transforms_.append((scale, ox, oy))
        batch = torch.stack(canvases).to(self.device)
        preds = self.cpm(batch, self.centermap.expand(len(boxes), -1, -1, -1))
        final = preds[-1].cpu()[:, 1:]                        # last stage, drop background
        coords = heatmap2coord(final).numpy()
        scores = final.flatten(2).max(dim=2).values.numpy()
        out = []
        for c, s, (scale, ox, oy) in zip(coords, scores, transforms_):
            c = c / scale                                     # 224-space -> crop-space
            c[:, 0] += ox                                     # crop-space -> image px
            c[:, 1] += oy
            out.append((c, s))
        return out

    def crop(self, image_rgb, box, coords, scores, align, min_score):
        """
        The classifier's input for one specimen.  Aligned: infer_unlabeled.py's box-mode
        align_crop.  Otherwise (alignment off, or kp4/kp8 below min_score): the same square
        framing with zero rotation, edge-filled -- never black padding.
        """
        x1, y1, x2, y2 = box
        size = self.align_size
        s = size * self.align_fill / max(x2 - x1, y2 - y1, 1e-6)
        centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        a, b = ALIGN_PAIR
        if align and min(scores[a], scores[b]) >= min_score:
            out = align_crop(image_rgb, coords[a], coords[b], size=size, scale=s, centre=centre)
            if out is not None:
                return out, True
        M = np.array([[s, 0.0, size / 2.0 - s * centre[0]],
                      [0.0, s, size / 2.0 - s * centre[1]]])
        return cv2.warpAffine(image_rgb, M, (size, size), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE), False

    @torch.no_grad()
    def classify(self, crops):
        """Top-2 (class id, probability) per crop, one batch."""
        if not crops:
            return []
        batch = torch.stack([self.cls_transform(Image.fromarray(c)) for c in crops]).to(self.device)
        probs = self.classifier(batch).softmax(dim=1).cpu()
        top = probs.topk(2, dim=1)
        return [list(zip(i.tolist(), p.tolist())) for p, i in zip(top.values, top.indices)]

    def analyze(self, image_rgb, conf, iou, min_score, align, mode=DETECT_RTDETR, progress=None):
        """
        Detect, keypoint, crop (aligned where possible) and classify every specimen.
        Returns a dict: image (RGB, contiguous), dets [(score, box)], n_added (auto mode),
        kps [(coords, scores)], crops [(crop, aligned)], preds [top-2 (class id, prob)].

        progress: optional callable(stage, message), called after each stage ("detect",
        "keypoints", "align", "classify") -- used by apps/mcp/server.py to report live steps.
        """
        report = progress or (lambda stage, message: None)
        image_rgb = np.ascontiguousarray(image_rgb[:, :, :3])
        dets, n_added = self.detect(image_rgb, conf, iou, mode)
        report("detect", f"{mode}: {len(dets)} specimen(s) found")
        kps = self.keypoints(image_rgb, [box for _, box in dets])
        report("keypoints", f"keypoints kp4-kp8 predicted for {len(kps)} specimen(s)")
        crops = [self.crop(image_rgb, box, c, s, align, min_score)
                 for (_, box), (c, s) in zip(dets, kps)]
        report("align", f"{sum(a for _, a in crops)} of {len(crops)} pose-aligned"
                        + ("" if align else " (alignment off)"))
        preds = self.classify([c for c, _ in crops])
        report("classify", f"classified {len(preds)} crop(s)")
        return {"image": image_rgb, "dets": dets, "n_added": n_added, "kps": kps,
                "crops": crops, "preds": preds}

    def run(self, image_rgb, conf, iou, min_score, align, show_names, mode=DETECT_RTDETR):
        """analyze() plus render(): what the app shows."""
        r = self.analyze(image_rgb, conf, iou, min_score, align, mode)
        return self.render(r, conf, min_score, align, show_names, mode)

    def render(self, r, conf, min_score, align, show_names, mode=DETECT_RTDETR):
        """The overlay, crop gallery, tables and status line for an analyze() result."""
        image_rgb, dets, n_added, kps, crops, preds = (
            r["image"], r["dets"], r["n_added"], r["kps"], r["crops"], r["preds"])

        vis = image_rgb.copy()
        h, w = vis.shape[:2]
        line = max(2, round(max(h, w) / 600))
        font = max(0.4, max(h, w) / 2000)
        r = max(3, round(max(h, w) / 300))   # keypoint dot radius, same for every specimen
        kp_rows, spec_rows, gallery = [], [], []
        for i, ((det_score, box), (coords, scores), (crop, aligned), top, colour) in enumerate(
                zip(dets, kps, crops, preds, palette(len(dets))), 1):
            (cls, prob), (cls2, prob2) = top
            name = self.class_names[cls]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, line)
            label = f"#{i} {name} {prob:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, line)
            ty = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 4
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), colour, -1)
            cv2.putText(vis, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font,
                        (0, 0, 0), line, cv2.LINE_AA)

            for kp, (x, y), s in zip(KP_NAMES, coords, scores):
                p = (int(round(x)), int(round(y)))
                cv2.circle(vis, p, r + 1, (0, 0, 0), -1 if s >= min_score else 2)
                cv2.circle(vis, p, r, colour, -1 if s >= min_score else 2)
                if show_names:
                    cv2.putText(vis, kp[2:], (p[0] + r + 2, p[1] + r), cv2.FONT_HERSHEY_SIMPLEX,
                                font * 0.8, colour, max(1, line - 1), cv2.LINE_AA)
                kp_rows.append([i, round(det_score, 3), kp, round(float(x), 1),
                                round(float(y), 1), round(float(s), 3)])

            a, b = ALIGN_PAIR
            spec_rows.append([i, name, round(prob, 3), f"{self.class_names[cls2]} ({prob2:.2f})",
                              "yes" if aligned else "no", round(det_score, 3),
                              round(float(scores[a]), 3), round(float(scores[b]), 3)])
            gallery.append((crop, f"#{i} {name} {prob:.2f} ({'aligned' if aligned else 'not aligned'})"))

        n_aligned = sum(a for _, a in crops)
        n_low = sum(1 for _, (c, s) in zip(dets, kps) if s.min() < min_score)
        res_note = (f" (detection {mode}: {n_added} small specimen(s) added from the "
                    f"{AUTO_HIGH} pass)" if mode == DETECT_AUTO else f" (detection {mode})")
        status = (f"{len(dets)} specimen box(es) at confidence >= {conf:.2f}{res_note}; "
                  f"{n_aligned} aligned before classification"
                  f"{'' if align else ' (alignment switched off)'}; "
                  f"{n_low} with at least one keypoint below {min_score:.2f} (drawn hollow). "
                  f"Classifier: {self.classifier_name}.")
        if not dets:
            status = f"No specimens detected at confidence >= {conf:.2f}. Try lowering it."
        return vis, gallery, spec_rows, kp_rows, status

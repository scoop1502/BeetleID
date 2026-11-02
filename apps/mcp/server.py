"""
MCP server for beetle identification, self-contained (no ArthropodModel needed).

For each specimen in a photo, BeetleID's own copy of the pipeline (beetleid/foundation: detect,
pose-align, name the order) finds it; specimens whose order is Coleoptera are passed to the beetle
model, which says SPB (southern pine beetle, Dendroctonus frontalis), Cleridae (checkered beetles)
or other_Coleoptera.

Tool
    identify_beetles   per specimen: order (foundation) and beetle class with probabilities,
                       box, counts; saves an overlay image

Run with any environment built from environment/requirements.txt:
    python apps/mcp/server.py                      # overlays go to runs/mcp/outputs
Like the foundation server, stdout is reserved for the MCP protocol; models load on first use.
"""

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import uuid
from collections import Counter
from typing import Literal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # BeetleID
sys.path.insert(0, ROOT)              # the beetleid package, without needing pip install

from mcp.server.fastmcp import FastMCP   # noqa: E402
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
COLEOPTERA = "Coleoptera"

mcp = FastMCP("beetle-id", log_level="WARNING")
_lock = threading.Lock()
_state = {}
_args = None


def emit(stage, message):
    """Live progress for the ArthropodModel web chat (same format as its server)."""
    if not _args or not _args.progress_file:
        return
    try:
        os.makedirs(os.path.dirname(_args.progress_file), exist_ok=True)
        with open(_args.progress_file, "a") as h:
            h.write(json.dumps({"t": round(time.time(), 3), "call": _state.get("call"),
                                "tool": "identify_beetles", "stage": stage, "message": message}) + "\n")
    except OSError:
        pass


def models():
    """BeetleID's pipeline + beetle classifier, loaded once."""
    with _lock:
        if "pipe" not in _state:
            emit("load", "loading the pipeline and the beetle model (first call only)")
            with contextlib.redirect_stdout(sys.stderr):
                from beetleid import BeetleClassifier
                from beetleid.foundation import pipeline as sp
                _state["pipe"] = sp.Pipeline(
                    sp.first_existing(sp.YOLO_WEIGHTS, "YOLO weights"), sp.RTDETR_WEIGHTS[0],
                    sp.first_existing(sp.CPM_CHECKPOINTS, "CPM checkpoint"),
                    sp.first_existing(sp.CLASSIFIERS, "classifier"),
                    sp.first_existing(sp.CLASS_MAPPINGS, "class mapping"), _args.device, 256, 0.95)
                _state["beetle"] = BeetleClassifier(_args.device)
        return _state["pipe"], _state["beetle"]


@mcp.tool()
def identify_beetles(image_path: str, only_beetles: bool = True,
                     detector: Literal["rtdetr", "yolo_auto"] = "rtdetr", confidence: float = 0.5) -> dict:
    """Identify beetles in a microscope photo as SPB (southern pine beetle, Dendroctonus
    frontalis; a destructive pine pest), Cleridae (checkered beetles; predators of bark beetles)
    or other_Coleoptera.  Use it after analyze_tray reports Coleoptera, or whenever the user asks
    about pine beetles, bark beetles or checkered beetles.

    image_path: absolute, or relative to the BeetleID folder.
    only_beetles: True = only specimens whose order is Coleoptera;
        False = run the beetle model on every specimen (useful if the order looks wrong).
    Results from photo setups unlike the training photos are less reliable; mention probabilities.
    """
    import numpy as np
    from PIL import Image
    _state["call"] = uuid.uuid4().hex[:8]
    started = time.time()
    emit("start", f"{os.path.basename(image_path)}")
    try:
        full = os.path.expanduser(image_path)
        full = full if os.path.isabs(full) else os.path.join(ROOT, full)
        if not os.path.isfile(full) or not full.lower().endswith(EXTS):
            raise ValueError(f"image not found or not an image: {image_path}")
        pipe, beetle = models()
        image = np.asarray(Image.open(full).convert("RGB"))
        from beetleid.foundation import pipeline as sp
        mode = sp.DETECT_RTDETR if detector == "rtdetr" else sp.DETECT_AUTO
        with _lock, contextlib.redirect_stdout(sys.stderr):
            r = pipe.analyze(image, confidence, 0.5, sp.MIN_SCORE, True, mode,
                             progress=lambda stage, msg: emit(stage, msg))
            specimens, vis = [], image.copy()
            for i, ((score, box), (crop, aligned), top) in enumerate(
                    zip(r["dets"], r["crops"], r["preds"]), 1):
                (c1, p1), _ = top
                order = pipe.class_names[c1]
                s = {"id": i, "order": order, "order_probability": round(p1, 3),
                     "box_xyxy": [int(round(v)) for v in box], "aligned": bool(aligned)}
                if not only_beetles or order == COLEOPTERA:
                    b = beetle.classify([crop])[0]
                    s.update(beetle_class=b["class"], beetle_probability=b["probability"],
                             beetle_probabilities=b["probabilities"])
                specimens.append(s)
        found = [s for s in specimens if "beetle_class" in s]
        emit("classify", f"beetle model on {len(found)} of {len(specimens)} specimen(s)")
        out = {"image": full, "n_specimens": len(specimens), "n_checked_by_beetle_model": len(found),
               "beetle_counts": dict(Counter(s["beetle_class"] for s in found).most_common()),
               "specimens": specimens}
        if _args.overlay_dir:
            import cv2
            for s in specimens:
                x1, y1, x2, y2 = s["box_xyxy"]
                label = (f"#{s['id']} {s['beetle_class']} {s['beetle_probability']:.2f}" if "beetle_class" in s
                         else f"#{s['id']} {s['order']} (not checked)")
                colour = (220, 40, 40) if s.get("beetle_class", "").startswith("SPB") else (40, 160, 40)
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, max(2, image.shape[1] // 600))
                cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.6, image.shape[1] / 2200), colour, 2, cv2.LINE_AA)
            os.makedirs(_args.overlay_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(full))[0].replace(" ", "_") + "_beetles.jpg"
            cv2.imwrite(os.path.join(_args.overlay_dir, name), vis[:, :, ::-1])
            out["overlay_path"] = os.path.join(_args.overlay_dir, name)
        emit("result", ", ".join(f"{n} {c}" for c, n in out["beetle_counts"].items()) or "no beetles checked")
    except Exception as e:
        emit("error", f"{type(e).__name__}: {e}")
        raise
    emit("done", f"finished in {time.time() - started:.1f} s")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlay-dir", default=os.path.join(ROOT, "runs", "mcp", "outputs"))
    parser.add_argument("--progress-file", default=None, help="the web chat's progress.jsonl, for live steps")
    parser.add_argument("--preload", action="store_true")
    _args = parser.parse_args()
    if _args.preload:
        models()
    mcp.run(transport="stdio")

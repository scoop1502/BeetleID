# Adding two beetle classes to the classifier: Cleridae and SPB

**Date:** 2026-10-28 · **Result:** the production classifier now recognises 23 classes (the 21
orders + Cleridae + southern pine beetle) with no loss on the original classes.

## Summary

| | Before (21 classes) | After (23 classes) |
|---|---|---|
| Cleridae, 7 val photos of 2 unseen specimens | n/a | **7 / 7 (100%)** |
| SPB *Dendroctonus frontalis*, 16 val photos of unseen specimens | n/a | **16 / 16 (100%)** |
| The 21 original classes, 2,157 val photos | 95.73% | **95.92%** (+0.2 pts, within seed-to-seed SD of ±0.39) |
| Original photos wrongly called Cleridae or SPB | n/a | **0** |
| All 2,180 val photos | n/a | 95.96% |
| Training time (RTX 3060) | 10 min (30 epochs from ImageNet) | **3 min 39 s** (10 epochs from the 21-class model) |

*Version 2.* A first run (v1, kept in `runs/classifier/resnet18_23cls/`) split the clerids by photo
number and put specimen 002252 on both sides; this run splits every class by specimen.  v1 scored
6/6, 16/16 and 95.32% on the original classes.

The model was extended, not retrained from scratch: everything it had learned was kept, and only
its final layer was rebuilt for 23 outputs.

## Data

| Class | Photos received | Usable after QC | Train (repeats) | Val |
|---|---|---|---|---|
| Cleridae (`~/Downloads/Clerid_Beetles`) | 27 (`PICA0100`-`0126`) | 27 | 20 (×20 = 400), specimens up to 002251 | 7 (`PICA0120`-`0126`), specimens 002252, 002253 |
| SPB (`~/Downloads/SPB_Photos`) | 83 (`PICA0017`-`0099`) | 79 | 63 (×7 = 441), specimens up to 002243 | 16 (`PICA0084`-`0099`), specimens 002244 onwards |

Single beetles, 1920×1080, photographed with a ruler and a printed specimen number; each specimen
has several photos.  The split is by **specimen** (read from the printed number at each boundary),
so no specimen is in both train and val; a split by photo would inflate the score.  The photos
are also sorted into `data/new_classes/split/{train_set,val_set,excluded}/<class>/`.  New training
photos are listed several times so ~60 photos are not drowned out by 12,245 existing ones; the
train/val lists accept repeated lines.

## What was done

**1. Photos into the project**: copied to `data/new_classes/photos/` (110 files, no name clashes).

**2. Alignment**, identical to how the existing training crops were made (YOLO boxes, CPM
keypoints, rotate kp4 -> kp8 vertical, 256-px crops):
```bash
.venv/bin/python tools/yolo_predict_boxes.py --input data/new_classes/photos --out data/new_classes/yolo_boxes.json --conf 0.4
.venv/bin/python tools/infer_unlabeled.py --input data/new_classes/photos --output data/new_classes/keypoints \
    --box-mode file --boxes data/new_classes/yolo_boxes.json --save-crops data/new_classes/aligned --align-fallback crop --no-vis
```

**3. Quality check by eye** (contact sheet of all crops).  All 27 clerid crops were good.  In 14
SPB photos YOLO had boxed the **printed specimen number** instead of the beetle (104 of 110 photos
had several detections: beetle, number card, ruler), and 2 more were badly framed.  These 16 were
redone with the colour-threshold box (`--box-mode auto`), which picks the dark beetle and not the
card; 4 photos (`PICA0017`-`0020`) where the beetle is a few pixels wide were dropped.  Final
check: all 106 crops show a beetle.

**4. Training folder** `data/finetune_23cls/` (60 MB): `aligned/` holds symbolic links to the
14,402 existing crops plus the 106 new ones; `train.txt` / `val.txt` are the originals plus the
new lines; `mapping.txt` is the original plus:
```
Class: Cleridae, Class id: 21, Class count: 27
Class: SPB_Dendroctonus_frontalis, Class id: 22, Class count: 79
```
Ids 0-20 are unchanged, which is what lets the old model's knowledge carry over.

**5. Training** from the production 21-class model:
```bash
PYTHONPATH=.:training .venv/bin/python training/classification/finetune.py \
    --model resnet18 --nb_classes 23 --finetune models/classifier_resnet18_aligned.pth \
    --data_path data/finetune_23cls --image_dir aligned/ \
    --epochs 10 --warmup_epochs 1 --batch_size 16 --drop_path 0.05 --seed 0 \
    --output_dir runs/classifier/resnet18_23cls_v2 --log_dir runs/classifier/resnet18_23cls_v2/log --num_workers 8
```
(`--finetune` pointed at the 21-class model, now `models/archive/classifier_resnet18_21cls.pth`.)
`finetune.py` (lines 265-279) loads the checkpoint, drops only `fc.weight` / `fc.bias` because their
shape changes from 21 to 23 outputs, initialises a new final layer and trains it at 10× the
learning rate of the rest.  Validation accuracy per epoch: 95.1, 94.6, 94.7, 95.0, 95.5, 95.5,
95.6, 95.3, 96.0, 96.0%.  The last epoch (`checkpoint-9.pth`) was kept.

**6. Evaluation** with the new per-class tool (finetune.py only prints overall accuracy):
```bash
.venv/bin/python tools/eval_classifier.py --checkpoint runs/classifier/resnet18_23cls_v2/checkpoint-9.pth \
    --data_path data/finetune_23cls --image_dir aligned --mapping data/finetune_23cls/mapping.txt
```
The original classes are as good as before (95.92% vs 95.73%; Coleoptera 94.1% -> 94.7%).  No
original val photo was predicted as Cleridae or SPB.  Confidence on the new val photos is 0.83-0.92
for most (label smoothing caps it near 0.9); lowest `PICA0124` 0.55 and `PICA0091` 0.72, both correct.

**7. Promoted to production** (reversible):
```bash
mv models/classifier_resnet18_aligned.pth models/archive/classifier_resnet18_21cls.pth   # move: the file is a hard link
cp models/class_mapping.txt models/archive/class_mapping_21cls.txt
cp runs/classifier/resnet18_23cls_v2/checkpoint-9.pth models/classifier_resnet18_aligned.pth
cp data/finetune_23cls/mapping.txt models/class_mapping.txt
```
The Gradio app, MCP server and web chat use the 23 classes with no code changes: the checkpoint
records `nb_classes` and the names come from `models/class_mapping.txt`.

**Undo:** move `models/archive/classifier_resnet18_21cls.pth` back to
`models/classifier_resnet18_aligned.pth` and `models/archive/class_mapping_21cls.txt` back to
`models/class_mapping.txt`.

## Files

| Path | What it is | In git |
|---|---|---|
| `training/classification/ADD_NEW_CLASSES.md` | the reusable step-by-step guide this run followed | yes |
| `tools/eval_classifier.py` | new: per-class accuracy and confusions for any checkpoint | yes |
| `models/classifier_resnet18_aligned.pth`, `models/class_mapping.txt` | the 23-class production model and names | weights no, mapping yes |
| `models/archive/` | the previous 21-class model and mapping | no |
| `data/new_classes/` | photos, YOLO boxes, keypoints, aligned crops, logs | no |
| `data/new_classes/split/` | the original photos sorted into `train_set/`, `val_set/`, `excluded/` | no |
| `data/finetune_23cls/` | training folder (links + new crops, lists, mapping; `*_v1.txt` = v1 lists) | no |
| `runs/classifier/resnet18_23cls_v2/` (+ `.log`) | this run: 10 epoch checkpoints (~1.3 GB; only `checkpoint-9.pth` is needed) | no |
| `runs/classifier/resnet18_23cls/` (+ `.log`) | v1 run (superseded) | no |

## Caveats (worth saying in the demo)

1. **Small validation sets.** 7 and 16 photos of 2 and about 4 specimens: one photo is 14% or 6%.
   "100%" means no errors on these photos, not a perfect classifier.
2. **Same photo sessions.** Train and val photos of each new class come from one imaging session
   (same lighting, background, number cards).  The model may partly rely on those cues; photos from
   other sessions or labs would test it properly.
3. **Mixed taxonomic levels.** Cleridae (a family) and SPB (a species) are beetles, alongside the
   order Coleoptera.  The model now means "other Coleoptera" by Coleoptera.  It did not confuse
   them here, but existing Coleoptera photos that are actually clerids or bark beetles would blur
   the classes.
4. **Detection on these photos.** The detectors also box the printed specimen number (it looks like
   a dark object on a light background).  On whole photos in the apps, expect an extra box on the
   card; the classification of the beetle's own box is what matters.
5. **Keypoints.** The CPM keypoint model was trained mostly on springtails, mites and Hemiptera; 31
   of the 106 new crops (8 Cleridae, 23 SPB) could not be pose-aligned and were used unrotated, as
   in the original data.

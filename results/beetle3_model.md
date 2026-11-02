# Beetle model (3 classes): data, split, results

**Date:** 2026-11-02 · **Model:** `models/beetle_classifier_resnet18.pth` (= `runs/beetle3/checkpoint-9.pth`)

## Design

A second stage after the detect -> align -> order pipeline.  Specimens whose order is Coleoptera
are passed to a ResNet-18 with three classes:
`other_Coleoptera`, `Cleridae`, `SPB_Dendroctonus_frontalis`.  Unlike a 2-class Cleridae/SPB
model it can say "a beetle, but neither of these".  It starts from the foundation's 21-order
classifier (which already knows arthropods); only the final layer is new.

## Data and split

| Class | Train | Val | Source |
|---|---|---|---|
| other_Coleoptera | 1,693 | 321 | `data/other_coleoptera`: the order dataset's Coleoptera crops, its train/val split (by photo) |
| Cleridae | 20 photos ×20 = 400 | 7 (2 specimens) | `data/new_classes`, split **by specimen**: 002252-002253 val only |
| SPB | 63 photos ×7 = 441 | 16 (3 specimens) | `data/new_classes`, split **by specimen**: 002244 onwards val only |

**Out-of-domain test (never used for training or model choice):** 15 SPB photos from a different
microscope / lighting (`~/Downloads/Photos/SPB1-15.png`): back-lit dark silhouettes on a lavender
or pink background, no label card or ruler; 16 specimen crops after detection.

## Training

10 epochs from the 21-order classifier, batch 16, seed 0 (~50 s on an RTX 3060).  Two variants:
**A** standard (random crops only), **B** with `--color_aug` (colour jitter + random greyscale, a
new opt-in flag in ArthropodModel's `finetune.py`).  Rule fixed before testing: keep B unless its
validation accuracy is more than 1 point below A's.  Both reached 99.71% -> **B's recipe is production.**
B was then retrained with BeetleID's own copy of the code and data (`training/build_and_train_beetle3.sh`
-> `runs/beetle3`): identical per-class validation results (321/321, 7/7, 15/16); that run is the
production model.

Finding while doing this: `finetune.py` silently ignored `--aa` / `--color_jitter`; its training
transform had random crops only, so no classifier in the project had seen colour or lighting
variation.  `--color_aug` is off by default, so earlier runs are unchanged.

## Results

| | A | **B (production)** |
|---|---|---|
| Validation, overall (344) | 99.71% | **99.71%** |
| other_Coleoptera (321) | 100% | **100%** |
| Cleridae (7, unseen specimens) | 7/7 | **7/7** |
| SPB (16, unseen specimens) | 15/16 (1 -> Cleridae) | **15/16** (1 -> Cleridae) |
| **Out-of-domain SPB, beetle model on every crop (16)** | **0/16** | **0/16** |
| P(SPB) on those crops | 0.01-0.15 | 0.03-0.16 |

On the out-of-domain photos every crop is called `other_Coleoptera`: a safe failure (it does not
invent Cleridae), but the model has not learned SPB beyond its one photo session.  The foundation
itself calls only 4 of these 16 specimens Coleoptera (9 Collembola, 3 Hymenoptera), which is why
BeetleID's own MCP tool offers `only_beetles=False` (check every specimen).  As a plug-in in
ArthropodModel only specimens classified Coleoptera reach it, so these photos would not.

## What would fix the out-of-domain gap

1. **SPB and Cleridae photos from more sessions and setups** (different microscopes, lighting,
   with and without cards), split by specimen; even 30-50 more specimens should help most.
2. Use the 15 new-microscope photos then as training data, and keep another new session as the test.
3. The foundation's order classifier needs the same treatment (it calls back-lit beetles Collembola).

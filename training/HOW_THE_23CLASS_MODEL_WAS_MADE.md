# Adding new classes on top of the existing classifier

Worked example: **Cleridae** (27 photos, `~/Downloads/Clerid_Beetles`) and **SPB**, southern pine
beetle *Dendroctonus frontalis* (83 photos, `~/Downloads/SPB_Photos`), added to the 21-order
ResNet-18 as classes 21 and 22.  All commands run from the project root on the lab workstation.

**Taxonomy note.** Both are beetles, and the existing classes are orders, including Coleoptera
(id 2).  The new model therefore learns "Cleridae", "SPB" and "other Coleoptera".  That works if
the existing Coleoptera photos contain no clerids or bark beetles; check with your PI.  The
alternative is a separate beetle-level classifier applied only to specimens called Coleoptera.

## 1. Bring the photos into the project
```bash
mkdir -p data/new_classes/photos
cp ~/Downloads/Clerid_Beetles/*.jpg ~/Downloads/SPB_Photos/*.jpg data/new_classes/photos/
```
Clerids are `PICA0100`-`PICA0126`, SPB `PICA0017`-`PICA0099`; no name clashes with the existing data.

## 2. Align them exactly as the training crops were made
```bash
.venv/bin/python tools/yolo_predict_boxes.py --input data/new_classes/photos \
    --out data/new_classes/yolo_boxes.json --conf 0.4
.venv/bin/python tools/infer_unlabeled.py --input data/new_classes/photos \
    --output data/new_classes/keypoints --box-mode file --boxes data/new_classes/yolo_boxes.json \
    --save-crops data/new_classes/aligned --align-fallback crop --no-vis
```

## 3. Check every crop by eye and fix the bad ones
Open `data/new_classes/aligned/` in the file browser.  In this set all clerid crops are fine, but
YOLO picked the **label card** instead of the beetle in 14 SPB photos and framed 2 badly.  Redo
those with the colour-threshold box, which ignores the card:
```bash
mkdir -p data/new_classes/redo
for n in 0023 0024 0026 0032 0033 0036 0037 0042 0043 0044 0045 0046 0047 0048 0068 0069; do
    cp data/new_classes/photos/PICA$n.jpg data/new_classes/redo/; done
.venv/bin/python tools/infer_unlabeled.py --input data/new_classes/redo \
    --output data/new_classes/keypoints_redo --box-mode auto \
    --save-crops data/new_classes/aligned --align-fallback crop --no-vis   # overwrites those crops
rm data/new_classes/aligned/PICA00{17,18,19,20}.jpg     # beetle too small to use in these 4
```
Look again: every crop should show a beetle.  That leaves 27 Cleridae and 79 SPB crops.

## 4. Build the training folder (old data + new data)
```bash
D=data/finetune_23cls
mkdir -p $D/aligned
cp -rs "$PWD/data/aligned/." $D/aligned/            # links to the 14,402 existing crops (no copies)
cp data/new_classes/aligned/*.jpg $D/aligned/
cp data/train.txt data/val.txt data/mapping.txt $D/

# class names: ids 0-20 must stay exactly as they are; new classes get 21 and 22
printf "Class: Cleridae, Class id: 21, Class count: 27\n"                    >> $D/mapping.txt
printf "Class: SPB_Dendroctonus_frontalis, Class id: 22, Class count: 79\n"  >> $D/mapping.txt

# split by SPECIMEN (the printed number on the card), never by photo: check the photos either side
# of each boundary.  Cleridae: specimens 002252-002253 = PICA0120-0126 go to val; SPB: 002244+ = PICA0084+
# and repeat the few new training photos so they are not drowned out by 12,000 others
for n in $(seq 100 119); do for i in $(seq 20); do printf "PICA%04d.jpg 21\n" $n; done; done >> $D/train.txt  # 20 x 20
for n in $(seq 120 126); do printf "PICA%04d.jpg 21\n" $n; done                              >> $D/val.txt    # 7
for n in $(seq 21 83);   do for i in $(seq 7);  do printf "PICA%04d.jpg 22\n" $n; done; done >> $D/train.txt  # 63 x 7
for n in $(seq 84 99);   do printf "PICA%04d.jpg 22\n" $n; done                              >> $D/val.txt    # 16
```

## 5. Train, starting from the existing 21-class model
```bash
PYTHONPATH=.:training .venv/bin/python training/classification/finetune.py \
    --model resnet18 --nb_classes 23 \
    --finetune models/classifier_resnet18_aligned.pth \
    --data_path data/finetune_23cls --image_dir aligned/ \
    --epochs 10 --warmup_epochs 1 --batch_size 16 --drop_path 0.05 --seed 0 \
    --output_dir runs/classifier/resnet18_23cls --log_dir runs/classifier/resnet18_23cls/log \
    --num_workers 8
```
The log should show `Removing key fc.weight / fc.bias`: everything learned is kept and only the
final layer is rebuilt for 23 classes.  The old model already knows the 21 orders, so 10 epochs
are enough.  Each epoch writes a ~130 MB checkpoint; keep the last (`checkpoint-9.pth`).

## 6. Evaluate per class
```bash
.venv/bin/python tools/eval_classifier.py --checkpoint runs/classifier/resnet18_23cls/checkpoint-9.pth \
    --data_path data/finetune_23cls --image_dir aligned --mapping data/finetune_23cls/mapping.txt
```
Look at: Cleridae and SPB accuracy (only 7 and 16 val photos, so one photo is 14% or 6%), whether
they are confused with Coleoptera, and whether overall accuracy stayed near the old 95.8%.

## 7. Make it the production model
`models/classifier_resnet18_aligned.pth` is a hard link to the old training run's checkpoint, so
**move it aside first**; copying on top of it would overwrite the old run too.
```bash
mkdir -p models/archive
mv models/classifier_resnet18_aligned.pth models/archive/classifier_resnet18_21cls.pth
cp models/class_mapping.txt models/archive/class_mapping_21cls.txt
cp runs/classifier/resnet18_23cls/checkpoint-9.pth models/classifier_resnet18_aligned.pth
cp data/finetune_23cls/mapping.txt models/class_mapping.txt
```
The apps and MCP server pick up 23 classes automatically (the checkpoint records `nb_classes`).
To undo, move the two archived files back.

## On the HPC instead
Same steps, with these differences: upload `data/new_classes/aligned/` and the lists from step 4;
the existing crops there are in `data/aligned_images_yolo/` (adjust the `cp -rs` line); run step 5
by copying `hpc/finetune_resnet.slurm` and changing `--nb_classes`, `--finetune`, `--data_path`,
`--image_dir`, `--epochs`, `--warmup_epochs` and the output folders as above.

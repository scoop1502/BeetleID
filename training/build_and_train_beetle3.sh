#!/usr/bin/env bash
# Rebuild the 3-class beetle dataset (other_Coleoptera / Cleridae / SPB) and train the beetle model.
# Self-contained: uses only BeetleID's files.  Run from the BeetleID folder with an environment
# built from environment/requirements.txt.  About 1 minute on an RTX 3060.
#     bash training/build_and_train_beetle3.sh [python]      # default python: python3
set -euo pipefail
PY="${1:-python3}"
D=data/beetle3
RUN=runs/beetle3
[ -e $D ] && { echo "$D exists; remove it first"; exit 1; }
[ -e $RUN ] && { echo "$RUN exists; remove it first"; exit 1; }
mkdir -p $D/aligned
# other_Coleoptera: 2,014 beetle crops (data/other_coleoptera, with their own train/val split)
for f in data/other_coleoptera/crops/*.jpg; do ln -s "$PWD/$f" $D/aligned/$(basename "$f"); done
for f in data/new_classes/aligned/*.jpg; do ln -s "$PWD/$f" $D/aligned/$(basename "$f"); done
printf "Class: other_Coleoptera, Class id: 0, Class count: 2014\nClass: Cleridae, Class id: 1, Class count: 27\nClass: SPB_Dendroctonus_frontalis, Class id: 2, Class count: 79\n" > $D/mapping.txt
awk '{print $1" 0"}' data/other_coleoptera/train.txt > $D/train.txt
awk '{print $1" 0"}' data/other_coleoptera/val.txt > $D/val.txt
# split by specimen (printed number): Cleridae 002252-3 and SPB 002244+ are validation only;
# training photos are repeated so ~80 photos are not drowned out by 1,693 other beetles
for n in $(seq 100 119); do for i in $(seq 20); do printf "PICA%04d.jpg 1\n" $n; done; done >> $D/train.txt
for n in $(seq 120 126); do printf "PICA%04d.jpg 1\n" $n; done >> $D/val.txt
for n in $(seq 21 83); do for i in $(seq 7); do printf "PICA%04d.jpg 2\n" $n; done; done >> $D/train.txt
for n in $(seq 84 99); do printf "PICA%04d.jpg 2\n" $n; done >> $D/val.txt
# start from the 21-order classifier (models/order_classifier_resnet18.pth);
# --color_aug = colour jitter + random greyscale (robustness to lighting / tint)
PYTHONPATH=training "$PY" training/classification/finetune.py \
    --model resnet18 --nb_classes 3 --finetune models/order_classifier_resnet18.pth \
    --data_path $D --image_dir aligned/ --epochs 10 --warmup_epochs 1 --batch_size 16 \
    --drop_path 0.05 --seed 0 --color_aug \
    --output_dir $RUN --log_dir $RUN/log --num_workers 8
echo "trained: $RUN/checkpoint-9.pth  (copy to models/beetle_classifier_resnet18.pth to use it)"

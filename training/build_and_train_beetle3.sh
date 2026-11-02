#!/usr/bin/env bash
# Rebuild the 3-class beetle dataset (other_Coleoptera / Cleridae / SPB) and train the model.
# Run from ~/Documents/BeetleID with the ArthropodModel environment.  ~1 minute on an RTX 3060.
set -euo pipefail
A=/home/nangbui/Documents/ArthropodModel
D=data/beetle3
[ -e $D ] && { echo "$D exists; remove it first"; exit 1; }
mkdir -p $D/aligned
# other_Coleoptera: the foundation's Coleoptera crops (order id 2), linked, with its own split
awk '$2==2{print $1}' $A/data/train.txt $A/data/val.txt | while read f; do ln -s $A/data/aligned/$f $D/aligned/$f; done
for f in data/new_classes/aligned/*.jpg; do ln -s "$PWD/$f" $D/aligned/$(basename $f); done
printf "Class: other_Coleoptera, Class id: 0, Class count: 2014\nClass: Cleridae, Class id: 1, Class count: 27\nClass: SPB_Dendroctonus_frontalis, Class id: 2, Class count: 79\n" > $D/mapping.txt
awk '$2==2{print $1" 0"}' $A/data/train.txt > $D/train.txt
awk '$2==2{print $1" 0"}' $A/data/val.txt > $D/val.txt
# split by specimen (printed number): Cleridae 002252-3 and SPB 002244+ are validation only;
# training photos are repeated so ~80 photos are not drowned out by 1,693 other beetles
for n in $(seq 100 119); do for i in $(seq 20); do printf "PICA%04d.jpg 1\n" $n; done; done >> $D/train.txt
for n in $(seq 120 126); do printf "PICA%04d.jpg 1\n" $n; done >> $D/val.txt
for n in $(seq 21 83); do for i in $(seq 7); do printf "PICA%04d.jpg 2\n" $n; done; done >> $D/train.txt
for n in $(seq 84 99); do printf "PICA%04d.jpg 2\n" $n; done >> $D/val.txt
# start from the foundation's 21-order classifier; --color_aug = colour jitter + greyscale
PYTHONPATH=$A:$A/training $A/.venv/bin/python $A/training/classification/finetune.py \
    --model resnet18 --nb_classes 3 --finetune $A/models/classifier_resnet18_aligned.pth \
    --data_path $D --image_dir aligned/ --epochs 10 --warmup_epochs 1 --batch_size 16 \
    --drop_path 0.05 --seed 0 --color_aug \
    --output_dir runs/beetle3_B --log_dir runs/beetle3_B/log --num_workers 8
cp runs/beetle3_B/checkpoint-9.pth models/beetle_classifier_resnet18.pth
cp $D/mapping.txt models/beetle_class_mapping.txt

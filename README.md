# BeetleID: beetle identification from microscope photos

Identifies beetles as

| id | class | what it is |
|---|---|---|
| 0 | `other_Coleoptera` | any other beetle |
| 1 | `Cleridae` | checkered beetles, predators of bark beetles |
| 2 | `SPB_Dendroctonus_frontalis` | southern pine beetle, a destructive pine pest |

**Self-contained.**  BeetleID has its own copy of everything it needs: the detection / pose-alignment /
order pipeline (`beetleid/foundation/`, copied from ArthropodModel commit 60b4094), the training
code, all model files and the training data.  It does not need ArthropodModel.

**Also a plug-in.**  ArthropodModel's MCP server uses `beetleid.BeetleClassifier` when this package is
installed in its environment: specimens it classifies as **Coleoptera** (and only those) get a
`beetle_id` in its results.  Install for that: `pip install -e ~/Documents/BeetleID --no-deps`.

## Layout

| Path | What it is | In git |
|---|---|---|
| `beetleid/classifier.py` | `BeetleClassifier`: beetle crops -> class + probabilities (the plug-in API) | yes |
| `beetleid/foundation/` | own copy of the pipeline: detect -> pose-align -> order (21 orders) | yes |
| `apps/mcp/server.py` | standalone MCP server, tool `identify_beetles` (beetles only by default) | yes |
| `training/build_and_train_beetle3.sh` | rebuilds the dataset and trains the beetle model (~1 min) | yes |
| `training/classification/`, `training/util/` | training code (`finetune.py`, with `--color_aug`) | yes |
| `training/HOW_THE_23CLASS_MODEL_WAS_MADE.md` | the earlier experiment: Cleridae/SPB added to the order classifier | yes |
| `environment/` | `requirements.txt` + setup notes (timm patch) | yes |
| `models/` | see below | mappings only |
| `results/beetle3_model.md` | the beetle model: data, split, results | yes |
| `results/new_classes_cleridae_spb.md`, `results/slides/` | the earlier 23-class experiment and its demo slides | yes |
| `data/new_classes/` | Cleridae and SPB photos, aligned crops, `split/{train_set,val_set,excluded}` | no |
| `data/other_coleoptera/` | 2,014 aligned beetle crops of other species + train/val lists | no |
| `data/beetle3/` | training folder built by the script (links to the two folders above) | no |
| `runs/` | training runs (`beetle3` = production) and the earlier experiments | no |

## Models (`models/`, ~670 MB)

| File | Use |
|---|---|
| `beetle_classifier_resnet18.pth`, `beetle_class_mapping.txt` | the beetle model (3 classes) = `runs/beetle3/checkpoint-9.pth` |
| `detector_rtdetr_v3.pt`, `detector_yolo11n.pt` | specimen detectors (RT-DETR default) |
| `keypoints_cpm_epoch240.pth` | keypoints for pose alignment |
| `order_classifier_resnet18.pth`, `order_class_mapping.txt` | 21-order classifier: finds the beetles, and the starting point for training |
| `class_mapping_23cls.txt` | class list of the earlier 23-class experiment |

## Use

```bash
# set up an environment once (see environment/README.md), then from this folder:
python apps/mcp/server.py                         # MCP server (stdio) for nanobot or another agent
bash training/build_and_train_beetle3.sh python   # retrain (remove data/beetle3 and runs/beetle3 first)

# from Python
import sys; sys.path.insert(0, "/path/to/BeetleID")
from beetleid import BeetleClassifier             # crops -> beetle classes
from beetleid.foundation import pipeline          # whole photos -> specimens, crops, orders
```

On this workstation the ArthropodModel environment has everything:
`~/Documents/ArthropodModel/.venv/bin/python`.

## Status and the main open problem

On held-out photos from the training session: other beetles 321/321, Cleridae 7/7, SPB 15/16.  On
**15 SPB photos from a different microscope** it recognises **none** (all other_Coleoptera), and the
order classifier calls most of them Collembola, so as a plug-in they never reach BeetleID.  All SPB
training photos come from one session; the fix is photos from several sessions and setups.
Details: `results/beetle3_model.md`.

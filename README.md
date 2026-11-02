# BeetleID: beetle identification on top of the ArthropodModel pipeline

ArthropodModel (`~/Documents/ArthropodModel`) is the foundation: it detects every specimen in a
microscope photo, pose-aligns it and names its **order** (21 orders).  BeetleID is a second stage
for beetles: for specimens the foundation calls **Coleoptera**, it says

| id | class | what it is |
|---|---|---|
| 0 | `other_Coleoptera` | any other beetle |
| 1 | `Cleridae` | checkered beetles, predators of bark beetles |
| 2 | `SPB_Dendroctonus_frontalis` | southern pine beetle, a destructive pine pest |

BeetleID does not copy the foundation's code: it uses the installed `arthropod` package (editable
install into `~/Documents/ArthropodModel/.venv`) and runs with that environment.

## Layout

| Path | What it is | In git |
|---|---|---|
| `apps/mcp/server.py` | MCP server with the `identify_beetles` tool (registered in nanobot as `beetle`) | yes |
| `training/build_and_train_beetle3.sh` | rebuilds the dataset and trains the beetle model (~1 min) | yes |
| `training/HOW_THE_23CLASS_MODEL_WAS_MADE.md` | the earlier experiment: Cleridae/SPB added to the order classifier | yes |
| `models/beetle_classifier_resnet18.pth`, `beetle_class_mapping.txt` | the beetle model and its classes | mapping only |
| `results/beetle3_model.md` | this model: data, split, results | yes |
| `results/new_classes_cleridae_spb.md`, `results/slides/` | the earlier 23-class experiment and its demo slides | yes |
| `data/new_classes/` | Cleridae and SPB photos, aligned crops, `split/{train_set,val_set,excluded}` | no |
| `data/beetle3/` | training folder (links to the foundation's Coleoptera crops + the new crops) | no |
| `runs/` | training runs (`beetle3_A`, `beetle3_B` = production; the 23-class runs) | no |

## Use

```bash
A=~/Documents/ArthropodModel
# MCP smoke test (or let nanobot / the ArthropodModel web chat call it)
$A/.venv/bin/python apps/mcp/server.py --help
# retrain
bash training/build_and_train_beetle3.sh
```

The ArthropodModel web chat (`apps/mcp/chat_app.py`) includes this server by default
(`--extra-mcp beetle`); its agent calls `identify_beetles` when a specimen looks like a beetle.

## Status and the main open problem

On held-out photos from the training session the model is near perfect (Cleridae 7/7, SPB 15/16,
other beetles 321/321).  On **15 SPB photos from a different microscope** it recognises **none**
(it calls them other_Coleoptera): all SPB training photos come from one session, and colour
augmentation alone does not bridge that.  The fix is SPB (and Cleridae) photos from several
sessions and setups.  Details: `results/beetle3_model.md`.

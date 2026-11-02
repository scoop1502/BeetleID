# Environments

| File | Where it is used |
|---|---|
| `requirements.txt` | the lab workstation's `.venv` (Python 3.10): pipeline, apps, detector training |
| `container_requirements.txt` | what is inside the HPC container `pytorch_latest.sif` (all `hpc/` jobs run in it) |
| `environment.yml` | an older conda environment from the original Insect-Foundation code (Windows) |

## Local setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r environment/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Required patch: timm 0.3.2 on PyTorch 2

`timm==0.3.2` (needed by the Insect-Foundation ViT code) imports `torch._six`, which PyTorch 2
removed.  After installing, run once:

```bash
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/' \
    .venv/lib/python3.10/site-packages/timm/models/layers/helpers.py
```

### nanobot (only for the AI-agent apps)

nanobot needs Python 3.11+, so it is installed separately with uv (no sudo needed); see
`apps/mcp/README.md`.

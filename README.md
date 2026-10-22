# multimodal-agent-rl

Research scaffold for text-guided industrial anomaly localization using a VLM,
multimodal RAG, crop/zoom tools, and future rLLM/GRPO post-training.
Crop/zoom tools, reproducibility/checkpoint helpers, and a Colab smoke notebook
are implemented. No datasets, pretrained models, or training runs are included.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BaharAzari/multimodal-agent-rl/blob/main/notebooks/colab_train.ipynb)

```text
src/multimodal_agent_rl/
  __init__.py
  tools.py              # crop_and_zoom
  runtime.py            # device, seeds, resumable checkpoints
tests/test_crop.py      # synthetic image tests
tests/test_runtime.py   # runtime/checkpoint tests
notebooks/colab_train.ipynb  # synthetic smoke run; no training
pyproject.toml          # dependencies and test/lint settings
uv.lock                 # reproducible dependency resolution
.python-version         # Python 3.13 development default
AGENTS.md               # coding conventions
PROJECT_PLAN.md         # architecture, milestones, experiments, metrics
```

## Local development

Use uv with an environment outside the project directory:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/multimodal-agent-rl-venv
uv sync --locked
uv run --locked pytest -p no:cacheprovider
uv run --locked ruff check --no-cache .
uv run --locked ruff format --check --no-cache .
```

Python >=3.11 is supported. Runtime dependencies are PyTorch, NumPy, and Pillow;
the dev group adds pytest and Ruff. Use `uv add PACKAGE` or `uv add --dev PACKAGE`
to update dependencies, and keep `pyproject.toml` and `uv.lock` together.
The environment is recreated on demand; `/tmp` may be cleared by the system.

## Google Colab

Open the notebook with the badge. Private repository access requires GitHub
authentication; alternatively, use an authenticated clone or upload an existing
checkout. Never put a token in a clone URL or a saved notebook cell.
Select **Runtime > Change runtime type > GPU** for CUDA; availability varies.
CPU fallback supports the synthetic tool/runtime smoke checks.

Use a fresh Python >=3.11 kernel; Colab does not need the local 3.13 development
pin. The notebook bootstraps `uv==0.11.13`, exports hashed runtime requirements
with `uv export --locked --no-dev --no-emit-project`, installs them into the active
kernel with `pip install --require-hashes`, then installs this `src/` package
with `pip install --no-deps -e .`. Do not run `uv sync` in Colab's preconfigured
kernel. Restart the runtime after dependency changes before continuing.

The notebook only checks synthetic images, device selection, seeding, and
checkpoint round trips. Its default `/content/checkpoints` is ephemeral. Optional
Drive mounting uses `/content/drive/MyDrive/multimodal-agent-rl/checkpoints` for
persistence. Run metadata captures the Git commit, `uv.lock` hash, versions, and
hardware. It does not download datasets/models or start training.

## Utilities

```python
from PIL import Image
from multimodal_agent_rl.tools import crop_and_zoom

image = Image.new("RGB", (100, 80), "gray")
view = crop_and_zoom(image, (10, 20, 40, 50), zoom_factor=2)
assert view.size == (60, 60)
```

Boxes are integer XYXY pixels, with exclusive right/bottom edges and strict image
bounds. Zoom must be finite and >=1. The utility preserves `L`, `RGB`, or `RGBA`
mode and leaves the input unchanged. See its docstring for rounding, bilinear
resampling, and mapping view coordinates back to the source image.

`multimodal_agent_rl.runtime` provides `select_device(require_cuda=False)`,
`seed_everything(seed=42)`, `save_checkpoint(path, *, model, optimizer=None,
step=0, metadata=None)`, and `load_checkpoint(path, *, model, optimizer=None,
map_location="cpu", restore_rng=True)`. Checkpoints store state dictionaries,
step/metadata, and RNG state, use atomic file replacement, and can restore RNGs
when resuming. These helpers do not implement a training loop.

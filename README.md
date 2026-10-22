# multimodal-agent-rl

Research scaffold for text-guided industrial anomaly localization using a VLM,
multimodal RAG, crop/zoom tools, and future rLLM/GRPO post-training.
Only `crop_and_zoom` is implemented. No data or models have been downloaded.

```text
src/multimodal_agent_rl/
  __init__.py
  tools.py              # crop_and_zoom
tests/test_crop.py      # synthetic image tests
pyproject.toml          # dependencies and test/lint settings
uv.lock                 # reproducible dependency resolution
.python-version         # Python 3.13 development default
AGENTS.md               # coding conventions
PROJECT_PLAN.md         # architecture, milestones, experiments, metrics
```

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

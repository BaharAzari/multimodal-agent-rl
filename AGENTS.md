# multimodal-agent-rl

Build a text-guided industrial anomaly-localization agent with multimodal RAG,
crop/zoom tools, PyTorch, and future rLLM/GRPO post-training. See `PROJECT_PLAN.md`.

## Scope and conventions

- Keep the layout minimal: `tools.py`, `runtime.py`, tests, and a Colab smoke
  notebook. Create future modules, `configs/`, or `experiments/` only as needed.
- Do not download datasets/models, build indexes, or train unless requested.
- Support Python >=3.11; `.python-version` pins 3.13 for development.
- Use small typed functions, public docstrings, explicit arguments, and Ruff.
  Avoid downloads, device allocation, or other work at import time.
- Use PyTorch for future ML components; isolate rLLM-specific integration.
- Keep notebooks thin: import the editable `src/` package; no duplicated tools or
  training logic. Current Colab cells use synthetic data and do not train.
- Reject unsupported types with `TypeError` and invalid values with `ValueError`.
- Boxes are original-image pixel XYXY with exclusive right/bottom edges.
  Require integer, nonempty, in-bounds crops; preserve caller-owned images.
- Future nested crops must retain offsets and actual resize scales so predictions
  map back to the original frame. Keep ground truth hidden from the policy.
- Seed Python/NumPy/PyTorch when used. Record config, code revision, `uv.lock`
  hash, uv/Python/model/data/index versions, seeds, hardware, and results.
- Index training data only; exclude held-out/duplicate images and label leakage.
  Tune on validation; compare methods under equal inference budgets.

## Dependencies and checks

Use uv and committed `uv.lock`. Keep environments outside the repository;
never create `.venv` unless requested. Prefix uv environment commands with
`UV_PROJECT_ENVIRONMENT=/tmp/multimodal-agent-rl-venv` (see `README.md`).
Use `uv sync --locked`, `uv add <name>` / `uv add --dev <name>`, and targeted
`uv lock --upgrade-package <name>`; review, test, and commit dependency changes.
For Colab's active Python >=3.11 kernel, use the notebook's pinned uv bootstrap,
locked hashed export, and kernel pip installs instead of `uv sync`. Do not force
the local Python pin. Restart after dependency changes; never save access tokens.

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/multimodal-agent-rl-venv
uv run --locked pytest -p no:cacheprovider
uv run --locked ruff check --no-cache .
uv run --locked ruff format --check --no-cache .
```

Keep tests offline, on CPU, and focused on synthetic fixtures and real boundaries.
Checkpoint changes must preserve state dictionaries, atomic writes, RNG restoration,
and step/metadata round trips. Keep optional Drive mounting explicit in notebooks.

# Research plan

## Objective and present scope

Given an industrial image and natural-language anomaly description, retrieve
relevant defect descriptions/reference images, inspect crops with a VLM, and
predict an original-image bounding box. Later improve localization through
verifiable rewards and GRPO using an isolated rLLM adapter.

Only the deterministic `crop_and_zoom` utility and tests are implemented.
No data/model downloads or training occur during setup. Keep the repository
minimal; create new modules, `configs/`, and `experiments/` only when implemented
work needs them. Use Python >=3.11, development Python 3.13, and PyTorch for ML.

## Proposed architecture

These are responsibilities for future implementation, not empty packages to add:

| Component | Responsibility |
| --- | --- |
| Data | Stable example IDs, images/descriptions/annotations, split manifests |
| Retrieval | Text/image embeddings, training-only indexes, ranked evidence/provenance |
| Agent | VLM observations, validated actions, history, bounded interaction loop |
| Tools | Crop/zoom now; coordinate/provenance wrapper when the agent is built |
| Rewards | Protected ground-truth scorer, localization and cost components |
| Training | Backend-neutral rollouts plus isolated rLLM/GRPO integration |
| Evaluation | Frozen protocols, metrics, ablations, and failure analysis |

An observation will contain the description, image, retrieved evidence, crop
history/coordinate frames, and remaining budget. Proposed actions are crop/zoom,
predict one box, abstain, or report no anomaly. A terminal action or fixed
step/token/tool limit ends the episode; malformed actions consume the budget.

The current tool accepts PIL `L`/`RGB`/`RGBA` images, integer pixel XYXY boxes,
and finite zoom >=1, then returns a bilinearly resized image. Right/bottom edges
are exclusive; boxes must satisfy `0 <= x1 < x2 <= width` and the analogous y rule.
A future wrapper must record original sizes, offsets, actual output dimensions,
and composed transforms for nested crops; the current tool returns no metadata.
A crop displayed at `(Wc, Hc)` maps `(u, v)` to source coordinates
`(x1 + u*(x2-x1)/Wc, y1 + v*(y2-y1)/Hc)`. Round only at the documented boundary.

## Milestones and acceptance criteria

| Stage | Work and acceptance criteria |
| --- | --- |
| 0. Setup | Minimal package, uv lock, crop/zoom; synthetic CPU tests and lint pass |
| 1. Data/contracts | Choose data, freeze splits; annotation and nested-coordinate round trips pass |
| 2. Direct VLM | Validated terminal outputs, fixed budgets, replayable evaluation baseline |
| 3. Retrieval | Training-only multimodal index; provenance and held-out leakage checks pass |
| 4. Tool agent | Bounded crop loop; all episodes terminate and boxes map to the original image |
| 5. Reward/rLLM | Scorer edge cases pass; synthetic multimodal rollout serializes and scores |
| 6. GRPO pilot | Controlled training; report held-out gains or failures and reward audits |
| 7. Study | Frozen ablations, repeated seeds, uncertainty, and error analysis |

Before pinning rLLM/backend versions, validate multimodal messages, crop handling,
tool trajectories, loss masks, rollout generation, KL/reference policy, and GRPO
support. Do not assume an API before this compatibility check.

## Research controls and experiments

- Freeze train/validation/test splits by source item/batch/duplicate group before
  indexing. Index training data only; exclude the query and derived duplicates.
  Keep held-out images, labels, and annotation-derived location hints out of RAG.
- Tune on validation and evaluate the frozen system on test. Record query/evidence
  provenance and define single-target annotation matching and normal-image cases.
- Compare direct VLM, +RAG, +tools, +both, and +GRPO with the same base model and
  declared image/token/tool/retrieval budgets. Use at least three fixed seeds;
  report actual consumption and training samples/rollouts/updates/compute.
- Ablate retrieval modality/top-k, crop count/scale, reward terms, and group size.
- Start with bounded terminal IoU reward; define normal/abstain/invalid scores
  explicitly. Add small bounded cost penalties only after the IoU-only baseline.
  Hide scorer state, validate finite boxes, cap actions, and audit reward exploits.
- GRPO groups sample the same input; document relative advantages, normalization,
  identical-reward groups, KL, reward variance, and valid completion rate.
- Report mean IoU, Acc@0.50/0.75 (`IoU >= threshold`), normal-case false positives,
  no-anomaly precision/recall, abstention coverage/selective accuracy, Recall@k,
  invalid/exhausted episodes, latency, tool calls, pixels/tokens, and cost.
  Count anomalous abstentions/invalid/exhausted outputs as primary-metric failures.
- Report per-category results, seed variation, paired changes, and uncertainty;
  inspect tiny defects, ambiguous text, irrelevant evidence, and frame errors.
- Reproduce dependencies with committed `uv.lock` and `uv sync --locked`; use
  `UV_PROJECT_ENVIRONMENT=/tmp/multimodal-agent-rl-venv` for environment commands.
  Keep `.venv` out of the repository. Record config/code/lock hashes, uv/Python
  and model/data/index versions, seeds/determinism, hardware, and artifact paths.
  Run the locked pytest/Ruff commands in `AGENTS.md` before comparing changes.

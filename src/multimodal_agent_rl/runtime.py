"""Device selection, reproducible seeds, and portable research checkpoints."""

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def select_device(require_cuda: bool = False) -> torch.device:
    """Prefer CUDA, then local MPS, then CPU; optionally require a CUDA GPU."""
    if not isinstance(require_cuda, bool):
        raise TypeError("require_cuda must be a boolean")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA is required but unavailable; select a GPU runtime")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch; backend/kernel determinism is separate."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not a boolean")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must lie in [0, 2**32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_components(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer | None
) -> None:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer or None")


def _validate_step(step: int) -> None:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer, not a boolean")
    if step < 0:
        raise ValueError("step must be nonnegative")


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save state_dicts, progress, JSON-compatible metadata, and RNGs.

    The temporary file shares the destination directory, and failed writes leave
    an existing checkpoint intact. Atomic rename on mounted Google Drive does
    not guarantee remote durability. No model objects or executable code are saved.
    """
    _validate_components(model, optimizer)
    _validate_step(step)
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary or None")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "metadata must contain finite JSON-compatible values"
        ) from error
    payload = {
        "schema_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "metadata": metadata if metadata is not None else {},
        "rng_state": _rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a trusted checkpoint into a matching model/optimizer and return payload.

    Uses ``weights_only=True`` and validates schema, parameter shapes, optimizer
    groups, and RNG states before applying them. Matching custom modules and
    optimizer implementations remain the caller's responsibility; application
    is not transactional if their load hooks fail. Omit optimizer for inference.
    CUDA RNGs are restored only with CUDA and the same GPU count; CPU RNGs remain
    portable. Scheduler, sampler, dataloader, and worker states are not included.
    ``restore_rng=False`` preserves the caller's current random streams.
    """
    _validate_components(model, optimizer)
    if not isinstance(restore_rng, bool):
        raise TypeError("restore_rng must be a boolean")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    required = {
        "schema_version",
        "model_state",
        "optimizer_state",
        "step",
        "metadata",
        "rng_state",
    }
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError("checkpoint payload is incomplete")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema_version")
    _validate_step(payload["step"])
    if not isinstance(payload["metadata"], dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    state = payload["model_state"]
    current = model.state_dict()
    if not isinstance(state, dict) or state.keys() != current.keys():
        raise ValueError("checkpoint model parameter keys do not match")
    for key, value in current.items():
        if isinstance(value, torch.Tensor) and (
            not isinstance(state[key], torch.Tensor) or state[key].shape != value.shape
        ):
            raise ValueError(f"checkpoint model parameter shape does not match: {key}")
    optimizer_state = payload["optimizer_state"]
    if optimizer is not None:
        if (
            not isinstance(optimizer_state, dict)
            or not isinstance(optimizer_state.get("state"), dict)
            or not isinstance(optimizer_state.get("param_groups"), list)
        ):
            raise ValueError("checkpoint has no valid optimizer state")
        groups = optimizer_state["param_groups"]
        if (
            any(
                not isinstance(group, dict) or not isinstance(group.get("params"), list)
                for group in groups
            )
            or len(groups) != len(optimizer.param_groups)
            or any(
                len(saved["params"]) != len(current["params"])
                for saved, current in zip(groups, optimizer.param_groups, strict=True)
            )
        ):
            raise ValueError("checkpoint optimizer parameter groups do not match")
    if restore_rng:
        try:
            rng = payload["rng_state"]
            numpy_state = (
                rng["numpy"][0],
                np.array(rng["numpy"][1], dtype=np.uint32),
                *rng["numpy"][2:],
            )
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(numpy_state)
            cpu_rng = rng["torch"].cpu()
            torch.Generator(device="cpu").set_state(cpu_rng)
            if not isinstance(rng["cuda"], list) or any(
                not isinstance(state, torch.Tensor)
                or state.dtype != torch.uint8
                or state.ndim != 1
                for state in rng["cuda"]
            ):
                raise ValueError("CUDA RNG states must be one-dimensional byte tensors")
        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            AttributeError,
            OverflowError,
        ) as error:
            raise ValueError("checkpoint RNG states are invalid") from error
    model.load_state_dict(state)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    if restore_rng:
        random.setstate(rng["python"])
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_rng)
        if torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return payload

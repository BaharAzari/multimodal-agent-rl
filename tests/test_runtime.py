"""Offline CPU checks for portable runtime and checkpoint utilities."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from multimodal_agent_rl.runtime import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_device_selection(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert select_device().type == expected


def test_required_cuda_fails_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        select_device(require_cuda=True)


def _draw_random_values() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(3)


def _assert_same_random_values(
    first: tuple[float, float, torch.Tensor], second: tuple[float, float, torch.Tensor]
) -> None:
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_seeds_repeat_across_python_numpy_and_torch() -> None:
    seed_everything(31)
    first = _draw_random_values()
    seed_everything(31)
    _assert_same_random_values(first, _draw_random_values())


@pytest.mark.parametrize("seed", [True, 1.0, "42", None])
def test_seed_rejects_invalid_types(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        seed_everything(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_rejects_values_outside_numpy_range(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must lie"):
        seed_everything(seed)


def test_checkpoint_restores_model_optimizer_progress_and_rng(tmp_path: Path) -> None:
    seed_everything(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    for parameter in model.parameters():
        optimizer.state[parameter]["momentum_buffer"] = torch.full_like(parameter, 0.5)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    path = save_checkpoint(
        tmp_path / "nested" / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=12,
        metadata={"run": "synthetic", "seed": 42},
    )
    expected_random = _draw_random_values()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer.param_groups[0]["lr"] = 99
    optimizer.state.clear()
    payload = load_checkpoint(path, model=model, optimizer=optimizer)

    assert payload["schema_version"] == 1
    assert payload["step"] == 12
    assert payload["metadata"] == {"run": "synthetic", "seed": 42}
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
    assert optimizer.param_groups[0]["lr"] == 0.2
    for parameter in model.parameters():
        assert torch.equal(
            optimizer.state[parameter]["momentum_buffer"],
            torch.full_like(parameter, 0.5),
        )
    _assert_same_random_values(expected_random, _draw_random_values())


def test_rng_restore_can_be_disabled(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=model)
    seed_everything(87)
    expected = _draw_random_values()
    seed_everything(87)
    load_checkpoint(path, model=model, restore_rng=False)
    _assert_same_random_values(expected, _draw_random_values())


def test_missing_optimizer_state_fails_before_model_mutation(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=source)
    target = torch.nn.Linear(2, 1)
    original = {key: value.clone() for key, value in target.state_dict().items()}
    optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="no valid optimizer state"):
        load_checkpoint(path, model=target, optimizer=optimizer)
    for key, value in target.state_dict().items():
        assert torch.equal(value, original[key])


def test_mismatched_model_fails_before_mutation(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=torch.nn.Linear(2, 1))
    target = torch.nn.Linear(3, 1)
    original = target.weight.detach().clone()
    with pytest.raises(ValueError, match="shape does not match"):
        load_checkpoint(path, model=target)
    assert torch.equal(target.weight, original)


@pytest.mark.parametrize("operation", ["save", "replace"])
def test_atomic_failure_preserves_existing_checkpoint_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=torch.nn.Linear(2, 1))
    original_bytes = path.read_bytes()

    def fail_operation(*args: object, **kwargs: object) -> None:
        raise OSError("simulated interrupted replacement")

    target = "torch.save" if operation == "save" else "os.replace"
    monkeypatch.setattr(f"multimodal_agent_rl.runtime.{target}", fail_operation)
    with pytest.raises(OSError, match="simulated interrupted"):
        save_checkpoint(path, model=torch.nn.Linear(2, 1), step=1)
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("corruption", ["optimizer", "cpu_rng", "cuda_rng"])
def test_invalid_state_fails_before_model_mutation(
    tmp_path: Path, corruption: str
) -> None:
    source = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    path = save_checkpoint(
        tmp_path / "checkpoint.pt", model=source, optimizer=optimizer
    )
    payload = torch.load(path, weights_only=True)
    if corruption == "optimizer":
        del payload["optimizer_state"]["state"]
    elif corruption == "cpu_rng":
        payload["rng_state"]["torch"] = torch.ones(2, dtype=torch.uint8)
    else:
        payload["rng_state"]["cuda"] = [0]
    torch.save(payload, path)
    target = torch.nn.Linear(2, 1)
    original = target.weight.detach().clone()
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="optimizer state|RNG states"):
        load_checkpoint(path, model=target, optimizer=target_optimizer)
    assert torch.equal(target.weight, original)


def test_unsupported_schema_is_rejected(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=model)
    payload = torch.load(path, weights_only=True)
    payload["schema_version"] = 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        load_checkpoint(path, model=model)


def test_cuda_rng_is_skipped_when_loading_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 1)
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=model)
    payload = torch.load(path, weights_only=True)
    payload["rng_state"]["cuda"] = [torch.get_rng_state()]
    torch.save(payload, path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_cuda_restore(states: list[torch.Tensor]) -> None:
        pytest.fail("CPU checkpoint loading must not restore CUDA states")

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", unexpected_cuda_restore)
    load_checkpoint(path, model=model)

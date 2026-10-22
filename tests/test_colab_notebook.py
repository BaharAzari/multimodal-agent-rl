"""Validate the notebook and execute its offline examples against src/."""

import ast
import json
import os
import subprocess
from functools import wraps
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks/colab_train.ipynb"


def test_notebook_is_valid_python_with_no_saved_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    imports = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        tree = ast.parse("".join(cell["source"]))
        imports.update(
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        if "smoke" in cell["metadata"]["tags"]:
            assert not any(
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                for node in ast.walk(tree)
            ), "Keep reusable implementation in src/, not notebook cells"
    assert "multimodal_agent_rl.runtime" in imports
    assert "multimodal_agent_rl.tools" in imports


def test_notebook_examples_run_offline_on_cpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        wraps(torch.backends.mps.is_available)(lambda: False),
    )
    monkeypatch.setenv("ANOMALY_RL_CHECKPOINT_DIR", str(tmp_path))
    namespace = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "Path": Path,
        "os": os,
        "subprocess": subprocess,
        "display": lambda *images: None,
    }
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if "smoke" in cell["metadata"].get("tags", []):
            exec(
                compile("".join(cell["source"]), f"{NOTEBOOK}:cell{index}", "exec"),
                namespace,
            )
    assert namespace["device"] == torch.device("cpu")
    assert namespace["checkpoint_path"] == tmp_path / "smoke.pt"
    assert namespace["checkpoint_path"].is_file()
    assert namespace["payload"]["metadata"]["git_commit"]
    assert namespace["view"].size == (60, 60)

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "mi-hulladekfelismero-train"
DATASET_VOLUME_NAME = "mi-hulladekfelismero-dataset"
MODELS_VOLUME_NAME = "mi-hulladekfelismero-models"

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATASET_DIR = LOCAL_PROJECT_ROOT / "dataset"
LOCAL_MODELS_DIR = LOCAL_PROJECT_ROOT / "models"

REMOTE_PROJECT_ROOT = PurePosixPath("/root/project")
REMOTE_DATASET_DIR = PurePosixPath("/mnt/dataset/dataset")
REMOTE_MODELS_DIR = PurePosixPath("/mnt/models")

app = modal.App(APP_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "src"),
        remote_path=str(REMOTE_PROJECT_ROOT / "src"),
    )
)

RESET = "\033[0m"
COLORS = {
    "info": "\033[96m",
    "step": "\033[94m",
    "ok": "\033[92m",
    "warn": "\033[93m",
    "error": "\033[91m",
    "train": "\033[95m",
    "eval": "\033[36m",
}


def log(message: str, kind: str = "info") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    color = COLORS.get(kind, "")
    print(f"{color}[{timestamp}] {message}{RESET}", flush=True)


def remote_runtime_path(path: PurePosixPath) -> Path:
    return Path(path.as_posix())


def run_command(command: list[str], cwd: Path, stream_kind: str) -> str:
    log(f"Running command in {cwd}: {' '.join(command)}", "step")
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output_lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        output_lines.append(raw_line)
        if line:
            log(line, stream_kind)

    return_code = process.wait()
    duration = time.perf_counter() - started
    if return_code != 0:
        log(f"Command failed after {duration:.1f}s with exit code {return_code}", "error")
        raise subprocess.CalledProcessError(return_code, command, output="".join(output_lines))

    log(f"Command finished successfully in {duration:.1f}s", "ok")
    return "".join(output_lines)


def train_impl(
    accelerator: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    unfreeze_all: bool,
) -> dict:
    log(f"Remote training job started on {accelerator.upper()}", "step")
    log(f"Run name: {run_name}", "info")
    log(
        "Hyperparameters: "
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"num_workers={num_workers}, unfreeze_all={unfreeze_all}",
        "info",
    )

    remote_project_root = remote_runtime_path(REMOTE_PROJECT_ROOT)
    remote_dataset_dir = remote_runtime_path(REMOTE_DATASET_DIR)
    remote_models_dir = remote_runtime_path(REMOTE_MODELS_DIR)

    checkpoint_dir = remote_models_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log(f"Checkpoint directory: {checkpoint_dir}", "info")

    train_command = [
        sys.executable,
        "src/train.py",
        "--data_dir",
        str(remote_dataset_dir),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--num_workers",
        str(num_workers),
        "--checkpoint_dir",
        str(checkpoint_dir),
    ]
    if unfreeze_all:
        train_command.append("--unfreeze_all")

    log("Starting remote training phase", "step")
    train_stdout = run_command(train_command, cwd=remote_project_root, stream_kind="train")

    model_path = checkpoint_dir / "best_model.pth"
    eval_command = [
        sys.executable,
        "src/evaluate.py",
        "--model_path",
        str(model_path),
        "--data_dir",
        str(remote_dataset_dir),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
    ]
    log(f"Starting remote evaluation phase using {model_path}", "step")
    eval_stdout = run_command(eval_command, cwd=remote_project_root, stream_kind="eval")

    (checkpoint_dir / "train.log").write_text(train_stdout, encoding="utf-8")
    (checkpoint_dir / "evaluation.txt").write_text(eval_stdout, encoding="utf-8")

    summary = {
        "accelerator": accelerator,
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_workers": num_workers,
        "unfreeze_all": unfreeze_all,
        "checkpoint_volume_path": f"{run_name}/best_model.pth",
        "evaluation_volume_path": f"{run_name}/evaluation.txt",
        "train_log_volume_path": f"{run_name}/train.log",
    }
    (checkpoint_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    log("Committing model artifacts to Modal Volume", "step")
    models_volume.commit()
    log(f"Remote job finished. Checkpoint stored at volume path {run_name}/best_model.pth", "ok")
    return summary


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    timeout=60 * 60,
    volumes={
        "/mnt/dataset": dataset_volume.read_only(),
        "/mnt/models": models_volume,
    },
)
def train_t4(
    run_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    unfreeze_all: bool,
) -> dict:
    return train_impl(
        accelerator="t4",
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        unfreeze_all=unfreeze_all,
    )


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    timeout=60 * 60,
    volumes={
        "/mnt/dataset": dataset_volume.read_only(),
        "/mnt/models": models_volume,
    },
)
def train_l4(
    run_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    unfreeze_all: bool,
) -> dict:
    return train_impl(
        accelerator="l4",
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        unfreeze_all=unfreeze_all,
    )


def sync_dataset_volume(force_overwrite: bool = True) -> None:
    if not LOCAL_DATASET_DIR.exists():
        raise FileNotFoundError(f"Local dataset folder not found: {LOCAL_DATASET_DIR}")

    log(f"Preparing dataset sync from {LOCAL_DATASET_DIR}", "step")
    try:
        dataset_volume.remove_file("dataset", recursive=True)
        log("Removed previous /dataset folder from the Modal dataset volume.", "warn")
    except Exception:
        log("No previous /dataset folder found in the Modal dataset volume.", "info")

    started = time.perf_counter()
    with dataset_volume.batch_upload(force=force_overwrite) as batch:
        batch.put_directory(str(LOCAL_DATASET_DIR), "/dataset")
    duration = time.perf_counter() - started
    log(
        f"Uploaded {LOCAL_DATASET_DIR} to the Modal dataset volume at /dataset in {duration:.1f}s.",
        "ok",
    )


def download_volume_file(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {remote_path} -> {local_path}", "step")
    with local_path.open("wb") as handle:
        for chunk in models_volume.read_file(remote_path):
            handle.write(chunk)
    log(f"Downloaded {local_path.name}", "ok")


def download_run_artifacts(run_name: str) -> None:
    local_run_dir = LOCAL_MODELS_DIR / run_name
    log(f"Downloading artifacts for run '{run_name}'", "step")
    download_volume_file(f"{run_name}/best_model.pth", local_run_dir / "best_model.pth")
    download_volume_file(f"{run_name}/evaluation.txt", local_run_dir / "evaluation.txt")
    download_volume_file(f"{run_name}/train.log", local_run_dir / "train.log")
    download_volume_file(f"{run_name}/summary.json", local_run_dir / "summary.json")
    log(f"Downloaded artifacts into {local_run_dir}", "ok")


@app.local_entrypoint()
def main(
    sync_dataset: bool = False,
    accelerator: str = "l4",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 4,
    run_name: str = "howa_modal_run1",
    unfreeze_all: bool = False,
    download_artifacts: bool = True,
) -> None:
    accelerator_name = accelerator.lower()
    if accelerator_name not in {"t4", "l4"}:
        raise ValueError("accelerator must be either 't4' or 'l4'")

    log("Modal training launcher started", "step")
    log(f"Requested accelerator: {accelerator_name.upper()}", "info")
    log(
        "Requested run: "
        f"run_name={run_name}, epochs={epochs}, batch_size={batch_size}, "
        f"lr={lr}, num_workers={num_workers}, sync_dataset={sync_dataset}, "
        f"download_artifacts={download_artifacts}",
        "info",
    )

    if sync_dataset:
        sync_dataset_volume()

    trainer = train_l4 if accelerator_name == "l4" else train_t4
    log(f"Submitting remote training job on {accelerator_name.upper()}", "step")
    summary = trainer.remote(
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        unfreeze_all=unfreeze_all,
    )

    log("Remote training summary:", "step")
    print(json.dumps(summary, indent=2), flush=True)

    if download_artifacts:
        download_run_artifacts(run_name)
    else:
        log("Artifact download skipped by configuration.", "warn")

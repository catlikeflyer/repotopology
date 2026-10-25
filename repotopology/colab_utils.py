"""Utilities for executing RepoTopology pipelines and training inside Google Colab."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


def is_colab() -> bool:
    """Returns True if code is executing in a Google Colab environment."""
    return "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Mounts Google Drive when running in Colab.

    Returns:
        True if Google Drive was successfully mounted or already available, False otherwise.
    """
    if not is_colab():
        print("[ColabUtils] Not running in Google Colab; skipping Google Drive mount.")
        return False

    try:
        from google.colab import drive  # type: ignore

        drive.mount(mount_point)
        print(f"[ColabUtils] Successfully mounted Google Drive at '{mount_point}'.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[ColabUtils] Note: Google Drive mount skipped or failed: {exc}")
        return False


def link_drive_data(
    drive_data_folder: str = "MyDrive/repology_data",
    mount_point: str = "/content/drive",
    target_data_dir: str | Path = "./data",
) -> Path:
    """Symlinks or links a Google Drive dataset directory to the local target_data_dir.

    This prevents copying large (20GB+) snapshot and graph directories into Colab's
    ephemeral disk. If the Drive directory does not exist, a local directory is created.
    """
    target = Path(target_data_dir).resolve()
    drive_path = Path(mount_point) / drive_data_folder

    if drive_path.exists():
        if target.is_symlink() or target.exists():
            if target.is_symlink():
                target.unlink()
            elif target.is_dir() and not any(target.iterdir()):
                target.rmdir()

        if not target.exists():
            target.symlink_to(drive_path, target_is_directory=True)
            print(f"[ColabUtils] Symlinked '{target}' -> '{drive_path}'. Zero local copy needed!")
            return target
        print(f"[ColabUtils] Target data dir '{target}' already exists; using existing directory.")
    else:
        target.mkdir(parents=True, exist_ok=True)
        print(
            f"[ColabUtils] Drive path '{drive_path}' not found yet. Using local directory '{target}'."
        )

    return target


def get_hardware_info() -> dict[str, Any]:
    """Inspects available GPU/TPU/CPU hardware in the environment."""
    info: dict[str, Any] = {
        "is_colab": is_colab(),
        "gpu_available": False,
        "device_name": "CPU",
        "total_memory_gb": 0.0,
        "bfloat16_supported": False,
    }

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_available"] = True
            info["device_name"] = props.name
            info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
            # Ampere or newer supports bf16
            info["bfloat16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        pass

    return info


def print_environment_summary() -> None:
    """Prints a formatted summary of the active Colab / local execution environment."""
    hw = get_hardware_info()
    colab_flag = "Google Colab" if hw["is_colab"] else "Local Machine"
    print("=" * 60)
    print(f"RepoTopology Environment Status: {colab_flag}")
    print(f"  GPU Available:      {hw['gpu_available']}")
    print(f"  Device Name:        {hw['device_name']}")
    if hw["gpu_available"]:
        print(f"  VRAM:               {hw['total_memory_gb']} GB")
        print(f"  bfloat16 Support:   {hw['bfloat16_supported']}")
    print("=" * 60)

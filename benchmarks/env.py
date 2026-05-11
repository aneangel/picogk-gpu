"""System info capture for benchmark reproducibility.

Records CPU, memory, GPU, and tool versions in each benchmark JSON so results
can be compared across hardware later.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def capture() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": {},
        "memory": {},
        "gpu": [],
        "tools": {},
    }

    cpu_model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass
    info["cpu"] = {"model": cpu_model, "logical_cores": _run(["nproc"])}

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["memory"]["total_kb"] = int(line.split()[1])
                    break
    except FileNotFoundError:
        pass

    if shutil.which("nvidia-smi"):
        gpu_csv = _run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        for line in gpu_csv.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                info["gpu"].append({
                    "name": parts[0],
                    "driver": parts[1],
                    "memory_mib": int(parts[2]) if parts[2].isdigit() else parts[2],
                    "compute_cap": parts[3],
                })

    for tool in ("dotnet", "cmake", "git"):
        if shutil.which(tool):
            v = _run([tool, "--version"])
            info["tools"][tool] = v.splitlines()[0] if v else ""
        else:
            info["tools"][tool] = None

    return info


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))

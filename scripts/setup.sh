#!/bin/bash
# Bootstrap script for picogk-gpu. Idempotent: safe to re-run.
# Pulls upstream Leap71 repos, installs apt build deps, installs .NET 9 SDK
# into ~/.dotnet, builds PicoGKRuntime for Linux, stages libpicogk.1.7.so
# so the harness host projects can load it via P/Invoke.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "==> picogk-gpu setup ($ROOT)"

clone_if_missing() {
  local url=$1 dest=$2
  if [[ -d "$dest/.git" ]]; then
    echo "  - $dest already present, skipping"
  else
    echo "  - cloning $url -> $dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

echo "==> [1/5] cloning upstream repos"
clone_if_missing https://github.com/leap71/PicoGK.git              third_party/PicoGK
clone_if_missing https://github.com/leap71/PicoGKRuntime.git       third_party/PicoGKRuntime
clone_if_missing https://github.com/leap71/LEAP71_ShapeKernel.git  third_party/LEAP71_ShapeKernel
clone_if_missing https://github.com/leap71/LEAP71_LatticeLibrary.git third_party/LEAP71_LatticeLibrary
clone_if_missing https://github.com/leap71/LEAP71_QuasiCrystals.git third_party/LEAP71_QuasiCrystals
clone_if_missing https://github.com/leap71/LEAP71_HelixHeatX.git benchmarks/helix_heatx/upstream
clone_if_missing https://github.com/leap71/LEAP71_RoverWheel.git benchmarks/rover_wheel/upstream

echo "==> [2/5] apt build deps (sudo)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  time build-essential pkg-config cmake \
  libtbb-dev libboost-all-dev libblosc-dev zlib1g-dev libjemalloc-dev \
  libgl1-mesa-dev xorg-dev libwayland-dev libxkbcommon-dev

echo "==> [3/5] .NET 9 SDK"
if [[ ! -x "$HOME/.dotnet/dotnet" ]]; then
  echo "  - installing .NET 9 SDK to ~/.dotnet"
  TMP=$(mktemp -d)
  curl -sSL https://dot.net/v1/dotnet-install.sh -o "$TMP/dotnet-install.sh"
  bash "$TMP/dotnet-install.sh" --channel 9.0 --install-dir "$HOME/.dotnet" --no-path
  rm -rf "$TMP"
else
  echo "  - ~/.dotnet/dotnet already present ($("$HOME/.dotnet/dotnet" --version))"
fi
if ! grep -q "/.dotnet" "$HOME/.bashrc" 2>/dev/null; then
  printf '\n# .NET 9 SDK (picogk-gpu)\nexport DOTNET_ROOT="$HOME/.dotnet"\nexport PATH="$DOTNET_ROOT:$PATH"\n' >> "$HOME/.bashrc"
  echo "  - appended dotnet PATH export to ~/.bashrc"
fi

echo "==> [4/5] init PicoGKRuntime submodules (OpenVDB, GLFW)"
( cd third_party/PicoGKRuntime && git submodule update --init --depth 1 --recursive )

echo "==> [5/5] build PicoGKRuntime + stage libpicogk.1.7.so"
if [[ ! -f "third_party/PicoGK/native/linux-x64/libpicogk.1.7.so" ]]; then
  ( cd third_party/PicoGKRuntime && \
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_BUILD_DOCS=OFF \
      -DOPENVDB_BUILD_UNITTESTS=OFF -DOPENVDB_BUILD_PYTHON_MODULE=OFF \
      -DOPENVDB_BUILD_AX=OFF \
      -DGLFW_BUILD_DOCS=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_EXAMPLES=OFF \
      -DUSE_BLOSC=ON -DUSE_ZLIB=ON
    cmake --build build -j --config Release
  )
  mkdir -p third_party/PicoGK/native/linux-x64
  cp third_party/PicoGKRuntime/build/lib/picogk.so \
     third_party/PicoGK/native/linux-x64/libpicogk.1.7.so
  echo "  - staged libpicogk.1.7.so"
else
  echo "  - libpicogk.1.7.so already staged, skipping build"
fi

echo
echo "==> done. quick sanity check:"
echo "  source ~/.bashrc"
echo "  python3 benchmarks/harness.py --name helix_heatx --project benchmarks/helix_heatx/Host --voxel-size 1.0 --runs 1"
echo "  python3 benchmarks/scoreboard.py"

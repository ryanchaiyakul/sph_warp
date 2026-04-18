# sph_warp

A GPU-accelerated Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH) fluid solver with rigid body coupling, built on [NVIDIA Warp](https://github.com/NVIDIA/warp) and integrated with [Newton](https://github.com/example/newton).

## Installation

### Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- Dependencies specified in `pyproject.toml`

### Setup

1. Clone the repository:
```bash
git clone git@github.com:ryanchaiyakul/sph_warp.git
cd sph_warp
```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

## Quick Start

### Running the Demo

The main demo simulates various rigid shapes (box, sphere, capsule, cylinder, cone, ellipsoid) dropping into a water container:

```bash
uv run main.py
```

This will:
1. Create a rigid body world with 6 shapes at drop height
2. Initialize a WCSPH fluid domain (3.5m × 1.0m × 1.0m)
3. Simulate fluid-rigid coupling for 360 frames at 60 FPS
4. Display real-time visualization with particles

## TODO:

- [ ] Parametrize fluid bouding region.
- [ ] Add rigid-rigid contact to `main.py`.
- [ ] Test with [NeRD](https://github.com/NVlabs/neural-robot-dynamics) as rigid solver.
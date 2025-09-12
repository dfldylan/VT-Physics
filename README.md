# VT-Physics

## 1. Introduction

An open-source physics engine for visual effects and research.

When conducting research tasks in the field of computer graphics, most papers
do not provide source code for actual testing and performance comparison. We developed
this library based on the actual research needs and practical engineering needs, and plan to
continuously improve the framework and implement as many classic techniques as possible.

### (1) Features

VT-Physics mainly focuses on the following aspects:

1. **Fluid Dynamics**: including particle-based, grid-based and hybrid fluid simulation
2. **Rigid-body Dynamics**
3. ...

More topics will be added in the future.

### (2) Implemented Techniques
1. "Position-Based Fluids" (**[2013-TOG. PBF](http://mmacklin.com/pbf_sig_preprint.pdf)**)
2. "Divergence-Free Smoothed Particle Hydrodynamics" (**[2015-SCA. DFSPH](https://dl.acm.org/doi/abs/10.1145/2786784.2786796)**)
3. "An Implicitly Stable Mixture Model for Dynamic Multi-fluid Simulations" (**[2023-SIGAsia. IMM](https://dl.acm.org/doi/abs/10.1145/2786784.2786796)**)
4. "Multiphase Viscoelastic Non-Newtonian Fluid Simulation" (**[2024-CGF. IMM-CT](https://dl.acm.org/doi/abs/10.1145/2786784.2786796)**)
5. ...

### (3) Recommended Build tool-chain

1. **CMake**: 3.10 or higher
2. **C++ & CUDA**:
    1. opt-1: Visual Studio 2019 with CUDA 11.6 or higher
    2. opt-2: Visual Studio 2022 with CUDA 12.6 or higher
3. **Platforms**: Windows / Linux.

## 2. Quick Start

### (1) Build

First, clone the project to your local directory:

```shell
git clone https://github.com/kevlns/VT-Physics.git
```

Then, run the bootstrap script to init the repo and install the dependencies:

```shell
cd VT-Physics
./bootstrap-vtphysics.bat   # for Windows
./bootstrap-vtphysics.sh    # for Linux
```

Finally, build the project with CMake:

**For Windows (with Visual Studio):**
```shell
cd VT-Physics
cmake -B build -S . -G "Visual Studio 16 2019"  # for CUDA 11.6+
# or
cmake -B build -S . -G "Visual Studio 17 2022"  # for CUDA 12.6+

# Build the project
cmake --build build --config Debug
```

**For Linux:**

First, ensure you have the necessary build tools installed. For Debian/Ubuntu based systems:
```shell
sudo apt update
sudo apt install build-essential curl zip unzip tar
```

Then, configure and build the project:
```shell
cd VT-Physics
cmake -B build -S .
cmake --build build --config Debug
```

### (2) Run

We provide simple examples associated with each physical solver. You can run the examples with a little modification to
test the solver. The examples are located in the `VT-Physics/Examples` directory.

For each solver, we provide a `README.md` file to introduce the solver and its usage, which you can find in the "VT-Physics/Simulator/Runtime/Include/Solvers" directory.

---

## 3. PBF Network Service (vp_pbf_server)

A lightweight TCP service process used to send/receive point clouds and control commands over the network, driving the PBF solver to advance frame by frame and returning particle data.

- Executable target: `vp_pbf_server` (built from `Simulator/CommandLineTool/PBFServer.cpp`)
- Dependencies: links against `vpmanager`; on Windows additionally links `Ws2_32`
- Default port: `55001`
- Run: `vp_pbf_server [port]`
- Connection model: single client, blocking I/O (after accepting one client it enters the loop)

### 3.1 Binary Protocol Overview

- Endianness: little-endian
- Every message starts with a 4‑byte signed int `type`, followed by the payload
- Basic scalar types: `int32`, `float32`
- Floating point arrays are contiguous flat `float32` arrays

Message types and payload / response:

1. Reset scene  
   - Request: `[int32 type=1][float particleRadius]`  
   - Action: clear all objects and solver, rebuild the PBF solver and update particle radius (other config uses a default template)  
   - Response: `[int32 ok]` (1 = success, 0 = failure)

1. Add Fluid point cloud  
   - Request: `[int32 type=2][int32 id][int32 N][float pos[N*3]][float vel[N*3]]`  
     - Current implementation ignores per‑particle velocity; all initial velocities set to 0  
   - Action: create `Particle_Common` with material FLUID, inject `pos` as particles, attach to PBF  
   - Response: `[int32 ok]`

1. Add Solid point cloud  
   - Request: `[int32 type=3][int32 id][int32 B][float pos[B*3]][float normal[B*3]]`  
     - `normal` is currently unused (reserved for future rigid/body boundary behavior)  
   - Action: create `Particle_Common` with material BOUNDARY; inject point cloud; attach to PBF  
   - Response: `[int32 ok]`

1. Next frame  
   - Request: `[int32 type=4][float dt]`  
     - If `dt > 0`, update PBF time step via a lightweight API (without resetting full config)  
   - Action: initialize solver lazily on first call; then advance one tick  
   - Response:  
     - Header: `[int32 ok=1][int32 objCount]`  
     - Per object (in the same order as attached), repeated `objCount` times:  
       - `[int32 id][int32 N][float pos[N*3]][float vel[N*3]]`  
       - `N` is particle count for that object (sliced via recorded start/end)

1. Clear all objects  
   - Request: `[int32 type=5]`  
   - Action: clear objects and solver; service keeps listening  
   - Response: `[int32 ok]`

1. Shutdown service  
   - Request: `[int32 type=9]`  
   - Action: close current connection and exit process  
   - Response: none (server closes socket)

### 3.2 Mapping Between Particles and Objects

- On every `attachObject`, the server records that object's `[start, end)` span inside the global particle array
- During a `Next frame` response, slices `pos/vel` for each object (in attachment order) and returns them

### 3.3 Key Implementation Details (for Integration / Extension)

- Direct raw point injection: `ParticleGeometryComponent::update` supports config key `__rawPoints__` (flat `std::vector<float>`), avoiding PLY files
- Lightweight data read-back added to `PBFSolver`:
  - `fetchAllParticles(std::vector<float3>& pos, std::vector<float3>& vel)`
  - `getAttachedObjectRanges(std::vector<int>& start, std::vector<int>& end)`
  - `setTimeStep(float dt)`
- Current limitations:
  - Single client, blocking I/O; no length-prefixed framing (relies on strict sender adherence)
  - Solid `normal` unused; `UpdateSolidTransform` message planned (apply transform and refresh GPU buffers before each tick)

### 3.4 Minimal Client Integration Notes

- Write `int32/float32` in little-endian
- Send a full message before sending the next to avoid interleaving (no custom framing)
- For `Next frame` response: first read `ok` and `objCount`, then iteratively read each object block

### 3.5 Planned (Not Yet Implemented)

- `UpdateSolidTransform` message (draft):
  - Request: `[int32 type=6][int32 id][float quat[4]][float trans[3]]`
  - Action: apply rigid transform to that solid's particle `pos` (and normals if used) on CUDA side before `tick`
  - Response: `[int32 ok]`
- Per-particle initial velocity upload for fluids
- Multi-client support / non-blocking I/O / heartbeat & timeout handling

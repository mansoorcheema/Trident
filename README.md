[//]: # (<p align="center">)

[//]: # (  <img src="docs/trident-logo.png" alt="Trident Logo" width="120" />)

[//]: # (</p>)

<h1 align="center">Trident — Lightweight inference, everywhere</h1>
<h3 align="center"> Inference Runtime for Edge AI</h3>
<br>

<p align="center">

[//]: # (  <a href="https://pypi.org/project/trident"><img src="https://img.shields.io/pypi/v/trident" alt="PyPI" /></a>)

[//]: # (  <a href="https://github.com/your-org/trident/actions"><img src="https://img.shields.io/github/actions/workflow/status/your-org/trident/ci.yml?branch=main" alt="Build Status" /></a>)

[//]: # (  <a href="https://github.com/your-org/trident/blob/main/LICENSE"><img src="https://img.shields.io/github/license/your-org/trident" alt="License" /></a>)

</p>


**Trident** is a C++ ML Inference Runtime with Python bindings for GPU accelerated Edge AI.  Built for large multi-modals on edge devices, robots, and anywhere you need performance without bloat.

It takes a trained neural network from **PyTorch**, **ONNX**, or **Tensorflow**, compiles it into a portable intermediate representation (IR), and executes it using parallelized **CPU** and **Vulkan** compute kernels.


## Why another inference runtime?

Workstation grade inference engines like TensorRT and ONNX Runtime are designed for large-scale workloads and server deployments. They excel in the data center but bring significant baggage to edge devices, mobile platforms, and developer workflows.

 **No dependency fights, no boilerplate, just drop it in and get inference running in three lines.**
 



## Features

- **Built for edge AI** – Brings multi-billion parameter models to Edge, matching or exceeding performance of ONNX in edge benchmarks.
- **Multi‑framework input** – import trained models directly from PyTorch, ONNX, and Tensorflow.
- **Hardware‑tuned kernels** – optimized for ARM NEON and Intel AVX2; register‑blocked, L1‑cache optimized; every kernel measured against hardware limits.
- **End-to-end GPU execution** – Handwritten GPU kernels for Neural Network Layers, keeps the entire graph on GPU with no CPU round trips or silent fallbacks.
- **Vulkan‑based GPU acceleration** –  runs on the GPUs edge hardware actually ships with: Adreno (Snapdragon), Mali (ARM), Intel, NVIDIA, and AMD.
- **Lightweight** – No TensorFlow runtime. No PyTorch runtime. No ONNX Runtime. No oneDNN, Eigen, or BLAS. Just a small C++ library, FlatBuffers, and Vulkan SDK headers.
- **Ahead‑of‑time compilation** – your model is compiled once into a self‑contained `.trident` IR file that can be deployed anywhere without the original framework.
- **Extensible** – register custom operators, define new layers, or plug in your own backend using simple C++.
- **Pythonic & fast** – first‑class Python bindings with zero‑copy NumPy / PyTorch tensor interchange.
- **Cross‑platform** – runs on Linux, Windows, macOS.




[//]: # (- **Vendor‑agnostic backends** – runs the same model on **CPU** and **Vulkan**, with **CUDA** support in development.)


## Benchmarks

Prefill latency for a 16-token prompt, fp32, versus ONNX Runtime on the same
machine and the same exported graph. Best of 15 runs (8 for Llama) after warm-up,
profiling disabled.

**SmolLM2-360M** 

| Threads | Trident | ONNX Runtime | |
|--------:|--------:|-------------:|:--|
| 1 | 41.2 ms | 41.1 ms | — |
| **4** | **13.9 ms** | 16.0 ms | **1.15× faster** |
| 8 | 26.4 ms | 23.0 ms | 0.87× |

**Llama-3.2-1B** 

| Threads | Trident | ONNX Runtime | |
|--------:|--------:|-------------:|:--|
| 1 | 368.5 ms | 365.8 ms | — |
| **4** | **113.9 ms** | 126.1 ms | **1.11× faster** |
| 8 | 225.1 ms | 164.0 ms | 0.73× |

**Peak memory usage**

| Model | Weights on disk | Trident | ONNX Runtime |
|-------|----------------:|--------:|-------------:|
| SmolLM2-360M | 0.54 GB | 0.74 GB | 0.72 GB |
| Llama-3.2-1B | 4.94 GB | 6.02 GB | 6.04 GB |

Weights are packed for the matmul kernel **in place at load time**. Model load is 2–3× faster than ORT including weight packing:
0.16 s vs 0.30 s for SmolLM2, 2.9 s vs 8.1 s for Llama.

Numerics are checked against ORT fp32 on every run: worst relative error
1.1e-05 (SmolLM2) and 7.8e-06 (Llama), with 100% top-1 agreement.

> **On 8 threads.** The test machine is an Apple M3 with 4 performance and 4
> efficiency cores. Trident's OpenMP loops use static scheduling, which hands
> every thread an equal share — so the four efficiency cores become stragglers
> that the fast cores wait on, and 8 threads is slower than 4. ORT's thread pool
> degrades more gracefully. **Use one thread per performance core.** Fixing this
> properly means asymmetry-aware partitioning, which is not implemented yet.

*Measured on Apple M3 (4 performance + 4 efficiency cores), 24 GB, macOS 15.7.3,
Clang -O3, with `OMP_WAIT_POLICY=active`
for Trident and `session.intra_op.allow_spinning=1` for ORT. Both runtimes execute
the same ONNX graph; Trident compiles it ahead of time with `-O`.*


## Quick Start

### Installation
```bash
# Clone and build
git clone https://github.com/mansoorcheema/trident-sdk.git
cd trident-sdk
cmake --preset release
cmake --build --preset release

# Python bindings
pip install trident  # local wheel

```

[//]: # (### From PyPI &#40;Python only&#41;)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install trident-sdk)

[//]: # ()
[//]: # (```)
### Run Inference

#### Python
```python
import trident as tri
import numpy as np

rt = tri.Runtime(backend="vulkan")
model = rt.load("bert.trident")

input_ids = np.random.randint(0, 30000, (1, 512)).astype(np.float32)
output = model.run(input_ids)
```

#### C++ 23
```C++
#include <trident/trident.hpp>
#include <vector>

auto ctx = trident::create_context(trident::Backend::Vulkan);
auto model = trident::model::load(*ctx, "bert.trident");

std::vector<float> input(512);
std::vector<float> output(model->output_size());

model->run(input, output);
```
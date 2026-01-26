# ⚡ RingTensor Extension (v1.3.1)

**RingTensor** is a high-performance, memory-resident C extension for the Ring programming language. It serves as the low-level mathematical engine for **RingML**, specifically optimized for Deep Learning, Transformers (GPT/BERT), and massive data processing.

> **Architecture:** RingTensor implements a **Hybrid Execution Model**. It utilizes **OpenMP** for Multi-Core CPU processing and **OpenCL** for GPU acceleration, automatically switching based on workload size.

## 🚀 What's New in v1.3.1
- **GPU Acceleration (OpenCL)**: Native support for Intel HD, NVIDIA, and AMD GPUs. Automatically offloads heavy matrix operations (`MatMul`, `Transpose`, `GELU`) to the GPU.
- **Binary Persistence**: Save and load tensors instantly using raw binary dumps (supports both Double-64bit and Float-32bit modes).
- **Advanced NLP Kernels**: Added **GELU** activation (GPT standard), **Batch Attention**, and Causal Masking.
- **Data Slicing**: Ultra-fast `memcpy`-based functions for slicing and inserting rows/columns (essential for Batching and Curriculum Learning).
- **Memory Optimization**: Improved `tensor_copy` and `tensor_set_from_list` to eliminate Ring interpreter overhead.

## ✨ Key Features
- **Zero-Copy**: Data resides entirely in C memory heaps. Ring handles lightweight pointers.
- **Smart Dispatcher**: The engine dynamically decides whether to use the CPU (for small/medium tasks) or GPU (for massive tensors) to minimize latency.
- **Double Precision**: Core math uses `double` (64-bit) for training stability.
- **Broadcasting**: Efficient row/vector broadcasting.

## 📦 Installation
```bash
ringpm install ringtensor from Azzeddine2017
```

## 🛠️ Build Instructions

### Windows (Visual Studio / MSVC)
To enable GPU support, ensure you have `OpenCL.lib` (or generate it using `lib.exe`).

```bat
cls
setlocal enableextensions enabledelayedexpansion
call ../../language/build/locatevc.bat x64

REM Build with OpenMP (CPU) and OpenCL (GPU) support
cl /c /O2 /Ot /GL /MD /openmp /DUSE_OPENCL ring_tensor.c -I"..\..\language\include" -I"./include"
link /LTCG /DLL ring_tensor.obj lib\OpenCL.lib ..\..\lib\ring.lib kernel32.lib /OUT:..\..\bin\ring_tensor.dll

del ring_tensor.obj
endlocal
```

### Linux / macOS (GCC)
```bash
gcc -shared -o libring_tensor.so -O3 -fPIC -fopenmp -DUSE_OPENCL ring_tensor.c -I ../../language/include -L ../../lib -lring -lOpenCL
```

---

## 📚 API Reference

### 1. Lifecycle, Shape & Persistence

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `tensor_init` | `rows, cols` | Allocates new tensor (0.0). Returns Pointer. |
| `tensor_copy` | `ptr` | Deep copy of a tensor (using `memcpy`). |
| `tensor_reshape` | `ptr, b, h, r, c` | Logically changes 4D dimensions. |
| `tensor_save` | `ptr, filename` | Saves data as Raw Binary (Double 64-bit). |
| `tensor_load` | `filename` | Loads data from Binary (Double 64-bit). |
| `tensor_save_fp32`| `ptr, filename` | Saves compressed (Float 32-bit). 50% smaller. |
| `tensor_load_fp32`| `filename` | Loads compressed (Float 32-bit) back to Double. |

### 2. High-Speed Data Manipulation
These functions use `memcpy` for instant data movement, critical for batching.

| Function | Description |
| :--- | :--- |
| `tensor_select_columns` | Copies specific columns to a new tensor. |
| `tensor_insert_columns` | Injects columns into a tensor. |
| `tensor_slice_rows` | Copies specific rows (Batch Slicing). |
| `tensor_insert_rows` | Injects rows into a tensor. |
| `tensor_set_from_list` | **Turbo Loader:** Fills tensor from a Ring List in one C pass. |
| `tensor_set_one_hot` | **Scatter:** Sets 1.0 at specific indices (for Targets). |

### 3. Matrix Operations (CPU + GPU Hybrid)

| Function | Description | GPU Support |
| :--- | :--- | :--- |
| `tensor_matmul` | `C = A * B` | ✅ Yes |
| `tensor_matmul_batch` | 3D Multiplication `[B,N,M]*[B,M,P]` | ❌ CPU (OpenMP) |
| `tensor_transpose` | `C = A.T` | ✅ Yes |
| `tensor_add_row_vec` | `A += Vec` (Broadcasting) | ❌ CPU (OpenMP) |
| `tensor_sum` | Sums rows or columns. | ❌ CPU (OpenMP) |

### 4. Activations

| Function | GPU Support | Notes |
| :--- | :--- | :--- |
| `tensor_gelu` | ✅ Yes | Gaussian Error Linear Unit (for GPT). |
| `tensor_relu` | ❌ CPU | Standard Rectified Linear Unit. |
| `tensor_sigmoid` | ❌ CPU | |
| `tensor_tanh` | ❌ CPU | |
| `tensor_softmax` | ❌ CPU | Stable implementation `exp(x-max)`. |

### 5. NLP & Transformer Kernels

| Function | Description |
| :--- | :--- |
| `tensor_embedding_forward` | Lookup Table (Index -> Vector). |
| `tensor_layernorm` | Layer Normalization (Mean/Var). |
| `tensor_attention_fast` | Fused Dot-Product Attention. |
| `tensor_attention_causal` | **Masked Attention:** Prevents looking at future tokens (GPT). |
| `tensor_attention_batch` | Processes full batch of attention heads in parallel. |

### 6. Optimizers

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `tensor_update_adam` | `W, G, M, V, lr, b1, b2, eps, t, wd` | Fused AdamW update with Weight Decay. |
| `tensor_update_sgd` | `W, G, lr` | Standard Stochastic Gradient Descent. |

---

## 💻 Usage Example: GPU Check

```ring
load "ringml.ring"

# 1. Check Capabilities
nCores = tensor_get_cores()
see "CPU Cores: " + nCores + nl

# 2. Configure
tensor_set_threads(2) # Optimize for dual-core

# 3. Large Matrix Operation
A = new Tensor(2000, 2000)
B = new Tensor(2000, 2000)
A.fill(1.5)
B.fill(2.5)

see "Running MatMul..."
t1 = clock()
C = A.matmul(B) # Will trigger GPU if available and size > threshold
see "Done in " + ((clock()-t1)/clockspersecond()) + "s"
```

## ⚠️ Important Notes
- **GPU Threshold**: The engine automatically falls back to CPU for small matrices to avoid PCIe transfer overhead.
- **Indexing**: API calls use 1-based indexing (Ring Standard). C internals use 0-based.
- **Persistence**: Always use `fp32` functions for deploying models to save disk space.

**Last Updated: 2026-01-25**

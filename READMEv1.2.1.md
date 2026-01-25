
# ⚡ RingTensor Extension (v1.2.1)

**RingTensor** is a high-performance, memory-resident C extension for the Ring programming language. It serves as the low-level mathematical backend for **RingML**, providing the speed required for Deep Learning, NLP, and Transformer-based models (like GPT).

> **Core Philosophy:** Zero-Copy Architecture. Data resides entirely in C memory (heaps); Ring only handles lightweight pointers. This eliminates data marshalling overhead, achieving near-C speeds.

## 🚀 New in v1.2.1
- **Multi-Core Processing (OpenMP)**: Automatically distributes heavy matrix operations across CPU cores.
- **4D Tensor Support**: Native support for logical reshaping `[Batch, Heads, Rows, Cols]`.
- **Batch Operations**: Specialized kernels for 3D Matrix Multiplication (`BatchMatMul`).
- **Memory Manipulation**: Blazing fast `memcpy`-based slicing and concatenation for Tensor batches and attention heads.

## ✨ Key Features
- **Double Precision**: All operations use `double` (64-bit) to ensure high accuracy for gradients.
- **Fused Kernels**: Optimizers (Adam, SGD) and Attention mechanisms calculate updates in a single C pass, bypassing interpreter overhead.
- **Broadcasting**: Efficient row/vector broadcasting (e.g., adding Bias to a Batch).
- **Transformer Ready**: Includes native kernels for Embedding, LayerNorm, and Causal Attention (GPT).
- **Stability**: Includes Numerically Stable Softmax and safe Division.

## 📦 Installation
```bash
ringpm install ringtensor from Azzeddine2017
```

## 🛠️ Build Instructions
To enable Multi-Core support, you **must** compile with OpenMP flags.

### Windows (Visual Studio / MSVC)
Create `buildvc.bat` in the extension folder:
```bat
cls
setlocal enableextensions enabledelayedexpansion
call ../../language/build/locatevc.bat x64

REM Build with /openmp for Multi-Threading support
cl /c /O2 /Ot /GL /MD /openmp ring_tensor.c -I"..\..\language\include"
link /LTCG /DLL ring_tensor.obj ..\..\lib\ring.lib kernel32.lib /OUT:..\..\bin\ring_tensor.dll

del ring_tensor.obj
endlocal
```

### Linux / macOS (GCC/Clang)
```bash
gcc -shared -o libring_tensor.so -O3 -fPIC -fopenmp ring_tensor.c -I ../../language/include -L ../../lib -lring
```


## 📚 API Reference

**Note:** All functions expect a **Managed Pointer** returned by `tensor_init`. 1-based indexing is used for API calls to match Ring standards.

### 1. Lifecycle & Shape Management

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `tensor_init` | `rows, cols` | Allocates a new tensor (initialized to 0.0). Returns Pointer. |
| `tensor_reshape` | `ptr, b, h, r, c` | Logically changes dimensions (4D) without moving data. Use `1` for unused dims. |
| `tensor_set` | `ptr, r, c, val` | Sets a specific value. |
| `tensor_get` | `ptr, r, c` | Gets a specific value. |

### 2. High-Performance Memory Ops (Slicing/Concat)
*Crucial for Batching and Multi-Head Attention.*

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `tensor_slice_rows` | `Src, Dest, StartRow, Count` | Copies a block of rows from Src to Dest using `memcpy`. |
| `tensor_insert_rows` | `Dest, Src, StartRow` | Inserts a block of rows from Src into Dest. |
| `tensor_select_columns` | `Src, Dest, StartCol, Count` | Copies specific columns (for splitting heads). |
| `tensor_insert_columns` | `Dest, Src, StartCol` | Inserts columns (for merging heads). |

### 3. Matrix Operations

| Function | Description | Notes |
| :--- | :--- | :--- |
| `tensor_matmul` | `C = A * B` | Highly optimized, multi-threaded matrix multiplication. |
| `tensor_matmul_batch` | `C = A * B` (3D) | Performs `[B, N, M] * [B, M, P]` in parallel over batches. |
| `tensor_transpose` | `C = A.T` | Optimized sequential write transposition. |
| `tensor_add_row_vec` | `A += Vec` | **Broadcasting:** Adds a vector to every row of the matrix (Bias Add). |
| `tensor_sum` | `A, Axis, Res` | `Axis=1`: Sum Rows (to Col). `Axis=0`: Sum Cols (to Row - used for Bias Grad). |
| `tensor_mean` | `A` | Returns the mean value of the entire tensor. |
| `tensor_argmax` | `A, Res` | Finds the index of the maximum value in each row. |

### 4. Transformer & NLP Kernels (The GPT Engine)

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `tensor_embedding_forward` | `Indices, Weights, Out` | High-speed Lookup Table. Converts integer IDs to Vectors. |
| `tensor_embedding_backward`| `Indices, GradOut, GradW`| Accumulates gradients for embeddings. |
| `tensor_layernorm` | `In, Gamma, Beta, Out, Eps`| Applies Layer Normalization (Mean/Var normalization + Scale/Shift). |
| `tensor_attention_fast` | `Q, K, V, Out, Scale` | **Fused Attention:** `Softmax(QK^T / s)V`. Efficient for Encoders. |
| `tensor_attention_causal` | `Q, K, V, Out, Scale` | **Masked Attention:** Applies `-inf` mask to future tokens. Essential for GPT training. |

### 5. Element-Wise Math (In-Place)

| Function | Logic | Function | Logic |
| :--- | :--- | :--- | :--- |
| `tensor_add` | `A += B` | `tensor_sub` | `A -= B` |
| `tensor_mul_elem` | `A *= B` | `tensor_div` | `A /= B` |
| `tensor_add_scalar` | `A += n` | `tensor_sub_scalar` | `A -= n` |
| `tensor_scalar_mul` | `A *= n` | `tensor_fill` | `A = n` |

### 6. Activation Functions
*All perform element-wise operations optimized with OpenMP.*

*   `tensor_sigmoid`, `tensor_sigmoid_prime`
*   `tensor_tanh`, `tensor_tanh_prime`
*   `tensor_relu`, `tensor_relu_prime`
*   `tensor_softmax`: Numerically stable Softmax (Row-wise).
*   `tensor_square`, `tensor_sqrt`, `tensor_exp`.

### 7. Optimizers (Fused)
Updates weights inside C to avoid interpreter loop latency.

*   **`tensor_update_adam(W, G, M, V, LR, B1, B2, Eps, T)`**: Full Adam implementation with Bias Correction.
*   **`tensor_update_sgd(W, G, LR)`**: Standard Stochastic Gradient Descent.
*   **`tensor_dropout(T, Rate)`**: Randomly zeros out elements for regularization.

### 8. System Utilities

| Function | Description |
| :--- | :--- |
| `tensor_get_cores` | Returns the number of logical processors available. |
| `tensor_set_threads` | Sets the number of OpenMP threads to use. (e.g. `tensor_set_threads(2)`). |

---

## 💻 Usage Example: Multi-Core Control

```ring
load "ringtensor.ring"

# 1. Check hardware capabilities
see "Detected Cores: " + tensor_get_cores() + nl

# 2. Performance Tuning
# For small matrices or dual-core CPUs, limiting threads can improve speed
# due to reduced overhead. For massive matrices, use max cores.
tensor_set_threads(2)

# 3. Create Tensors
pA = tensor_init(1000, 1000)
pB = tensor_init(1000, 1000)
pC = tensor_init(1000, 1000)

tensor_random(pA)
tensor_random(pB)

# 4. Matrix Multiplication (Parallelized)
t1 = clock()
tensor_matmul(pA, pB, pC)
see "Time: " + (clock()-t1)/clockspersecond() + "s" + nl
```
---

## 🧠 Core Concepts & Examples

### 1. The Basics: Creating & Manipulating Tensors
All functions return or accept a **Pointer** (`tensor_t*`), not a List.

```ring
load "ringtensor.ring"

# 1. Allocation (Rows, Cols) - Returns a Pointer
pA = tensor_init(3, 3) 
pB = tensor_init(3, 3)

# 2. Fill Data
tensor_fill(pA, 1.5)        # Fill all with 1.5
tensor_random(pB)           # Random [0, 1]

# 3. Element-wise Math (In-Place)
# pA = pA + pB
tensor_add(pA, pB)

# 4. Access Data (1-based index)
val = tensor_get(pA, 1, 1)
see "Value at (1,1): " + val + nl
```

### 2. Building a Neural Layer (Dense)
This example simulates `Output = Activation( Input x Weights + Bias )`.

```ring
# Input: Batch of 32 samples, 64 features
pInput = tensor_init(32, 64)
tensor_random(pInput)

# Weights: 64 inputs -> 128 neurons
pWeights = tensor_init(64, 128)
tensor_random(pWeights)

# Bias: 128 neurons (Row Vector)
pBias = tensor_init(1, 128)
tensor_random(pBias)

# Output Container
pOutput = tensor_init(32, 128)

# --- The Forward Pass ---

# 1. Matrix Multiplication (Heavy Lifting)
# Uses OpenMP to distribute 32 samples across cores
tensor_matmul(pInput, pWeights, pOutput)

# 2. Broadcasting (Add Bias to every row)
tensor_add_row_vec(pOutput, pBias)

# 3. Activation (ReLU)
tensor_relu(pOutput)

see "Forward Pass Complete." + nl
```

### 3. The GPT Engine (NLP Kernels)
RingTensor v1.2.1 introduces specialized kernels for Transformer models (like BERT and GPT).

#### A. Embedding Lookup
Converts integer tokens to dense vectors instantly.
```ring
# Vocabulary: 1000 words, Dim: 50
pEmbedWeights = tensor_init(1000, 50) 

# Input Sentence: [5, 20, 99] (Indices)
pIndices = tensor_init(1, 3)
tensor_set(pIndices, 1, 1, 5)
tensor_set(pIndices, 1, 2, 20)
# ...

# Output Container
pVectors = tensor_init(3, 50)

# Fast Lookup
tensor_embedding_forward(pIndices, pEmbedWeights, pVectors)
```

#### B. Causal Attention (The "Brain" of GPT)
Performs `Softmax( Q.K^T / scale ) . V` with masking to prevent looking at future tokens.

```ring
# Context: Sequence Length 10, Embedding Dim 64
pQ = tensor_init(10, 64)
pK = tensor_init(10, 64)
pV = tensor_init(10, 64)
pOut = tensor_init(10, 64)

# Run Fused Kernel
# Automatically handles Matrix Mul, Scaling, Masking, Softmax, and Context Mixing
tensor_attention_causal(pQ, pK, pV, pOut, 0.125)
```

---

#### ⚡ Performance Tuning (Multi-Threading)

RingTensor uses **OpenMP** to parallelize operations. However, for small matrices, the overhead of creating threads might be slower than serial execution.

```ring
# 1. Check Hardware
nCores = tensor_get_cores()
see "CPU Cores Detected: " + nCores + nl

# 2. Optimize
# If you are training small networks or using a dual-core CPU with Hyperthreading,
# limiting threads often yields better performance.
tensor_set_threads(2) 
```

---

## ⚠️ Important Notes
- **Memory Management**: Pointers returned by `tensor_init` are Managed Pointers. Ring's Garbage Collector will automatically call `free()` when the variable goes out of scope. You do not need to free them manually.
- **Dimensions**: Ensure dimensions match for operations like `add` or `matmul`, otherwise the extension may throw a runtime error.
- **Zero-Based vs One-Based**: Internally C uses 0-based indexing, but the API (`tensor_set`/`tensor_get`) uses 1-based indexing to match Ring's standard.
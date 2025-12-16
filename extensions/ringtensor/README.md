# ⚡ RingTensor Extension
RingTensor is a high-performance C extension for the Ring programming language, designed specifically to accelerate Deep Learning and Matrix operations. It provides a robust, double-precision mathematical engine that powers the RingML library.

> **Architecture Note:** Unlike standard Ring extensions that operate on Lists, RingTensor uses Memory-Resident Managed Pointers. This enables Zero-Copy arithmetic, making it up to 100x faster for heavy workloads.

## 🚀 Features
- **Double Precision**: All operations use double (64-bit) to ensure high accuracy for gradients.
- **Zero-Copy Architecture**: Data stays in C memory. Ring only holds pointers. No expensive marshalling between Ring and C.
- **Fused Kernels**: Optimizers (Adam, SGD) calculate updates in a single C pass, bypassing interpreter overhead.
- **Stability**: Includes Numerically Stable Softmax and safe Division.
## 📦 Installation
```bash
ringpm install ringtensor from Azzeddine2017
```

## 🛠️ Build Instructions
To get the maximum performance, we build in Release Mode with full optimizations.

### Windows (Visual Studio)
Create buildvc.bat in the extension folder:
```bat
cls
setlocal enableextensions enabledelayedexpansion
call ../../language/build/locatevc.bat x64

REM Build with Max Speed (/O2) and Link Time Code Generation (/LTCG)
cl /c /O2 /Ot /GL /MD ring_tensor.c -I"..\..\language\include"
link /LTCG /DLL ring_tensor.obj  ..\..\lib\ring.lib kernel32.lib /OUT:..\..\bin\ring_tensor.dll

del ring_tensor.obj
endlocal
```
### Linux / macOS (GCC)
```bash
gcc -shared -o libring_tensor.so -O3 -fPIC ring_tensor.c -I ../../language/include -L ../../lib -lring
```
## 📚 API Reference
Note: All functions expect Pointers returned by `tensor_init`, not Ring Lists.

### 1. Lifecycle & Access

| Function | Parameters | Return | Description |
| :--- | :--- | :--- | :--- |
| `tensor_init` | Rows, Cols | Pointer | Allocates memory for a new tensor (initialized to 0.0). |
| `tensor_set` | Ptr, Row, Col, Val | - | Sets a value at (Row, Col). 1-based indexing. |
| `tensor_get` | Ptr, Row, Col | Number | Gets a value from (Row, Col). |

### 2. Element-Wise Math (In-Place)
Operations modify the first tensor (A).

| Function | Parameters | Logic |
| :--- | :--- | :--- |
| `tensor_add` | Ptr A, Ptr B | A += B |
| `tensor_sub` | Ptr A, Ptr B | A -= B |
| `tensor_mul_elem` | Ptr A, Ptr B | A *= B (Hadamard) |
| `tensor_div` | Ptr A, Ptr B | A /= B |
| `tensor_scalar_mul` | Ptr A, Number n | A *= n |
| `tensor_add_scalar` | Ptr A, Number n | A += n |

### 3. Matrix Operations

| Function | Parameters | Description | Behavior |
| :--- | :--- | :--- | :--- |
| `tensor_matmul` | Ptr A, Ptr B, Ptr Res | Dot Product (A x B). | Writes to Res. |
| `tensor_transpose` | Ptr A, Ptr Res | Transposes A. | Writes to Res. |
| `tensor_sum` | Ptr A, Axis, Ptr Res | 1=Rows, 0=Cols. | Writes to Res. |
| `tensor_mean` | Ptr A | Mean of all items. | Returns Number. |
| `tensor_argmax` | Ptr A, Ptr Res | Max index per row. | Writes to Res. |

### 4. Transformations & Activations (In-Place)

| Function | Formula |
| :--- | :--- |
| `tensor_fill` | Fills with value n. |
| `tensor_random` | Fills with 0.0 to 1.0. |
| `tensor_square` | x^2 |
| `tensor_sqrt` | sqrt(x) |
| `tensor_exp` | e^x |
| `tensor_sigmoid` | 1 / (1 + e^-x) |
| `tensor_tanh` | tanh(x) |
| `tensor_relu` | max(0, x) |
| `tensor_softmax` | Stable Softmax (Exp-Normalize). |
### 5. Optimizers (Fused Kernels)
High-performance updates that happen entirely in C.

**`tensor_update_sgd`**
```ring
tensor_update_sgd(Ptr W, Ptr Grad, Number LR)
```
**`tensor_update_adam`**
```ring
tensor_update_adam(Ptr W, Ptr G, Ptr M, Ptr V, LR, Beta1, Beta2, Eps, T)
```
**`tensor_dropout`**
```ring
tensor_dropout(Ptr A, Number Rate)
```
## 💻 Usage Example
```ring
loadlib("ring_tensor.dll") # or .so

# 1. Create Tensors
pA = tensor_init(2, 2)
pB = tensor_init(2, 2)
pC = tensor_init(2, 2)

# 2. Set Values
tensor_fill(pA, 1.0)       # A = [[1,1],[1,1]]
tensor_set(pB, 1, 1, 5.0)  # B = [[5,0],[0,0]]

# 3. Math (A = A + B)
tensor_add(pA, pB)         # A becomes [[6,1],[1,1]]

# 4. Matrix Multiplication (C = A * B)
tensor_matmul(pA, pB, pC)

# 5. Print Result
see "Result (1,1): " + tensor_get(pC, 1, 1) + nl
```
## ⚠️ Important Notes
- **Memory Management**: Pointers returned by `tensor_init` are Managed Pointers. Ring's Garbage Collector will automatically call `free()` when the variable goes out of scope. You do not need to free them manually.
- **Dimensions**: Ensure dimensions match for operations like `add` or `matmul`, otherwise the extension may throw a runtime error.
- **Zero-Based vs One-Based**: Internally C uses 0-based indexing, but the API (`tensor_set`/`tensor_get`) uses 1-based indexing to match Ring's standard.
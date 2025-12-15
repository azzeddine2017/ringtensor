# ⚡ RingTensor Extension

RingTensor is a high-performance C extension for the Ring programming language, designed specifically to accelerate Deep Learning and Matrix operations. It provides a robust, double-precision mathematical engine that powers the RingML library.

## 🚀 Features

*   **Double Precision**: All operations use `double` to ensure high accuracy for gradients and learning rates.
*   **Fused Kernels**: Optimizers (Adam, SGD) are implemented as single C calls, eliminating thousands of interpreter loops.
*   **Stability**: Includes Numerically Stable Softmax and safe Division.
*   **Zero-Copy Logic**: Most element-wise operations work "In-Place" to minimize memory allocation overhead.

## 📦 Install

```bash
ringpm install ringtensor from Azzeddine2017
```

## 📦 Build Instructions

### Windows (Visual Studio)

Create `buildvc.bat` in the extension folder:

```bat
cls
cl /c /DEBUG ring_tensor.c -I"..\..\include"
link /DEBUG ring_tensor.obj  ..\..\lib\ring.lib /DLL /OUT:ring_tensor.dll
del ring_tensor.obj
```

Run `buildvc.bat`.

### Linux / macOS (GCC)

```bash
gcc -shared -o libring_tensor.so -fPIC ring_tensor.c -I ../../include -L ../../lib -lring
```

## 📚 API Reference

### 1. Matrix Initialization & Scalars

| Function | Parameters | Description | Behavior |
| :--- | :--- | :--- | :--- |
| `tensor_fill` | List A, Number n | Fills matrix with value n. | In-Place |
| `tensor_random` | List A | Fills matrix with uniform randoms 0.0 to 1.0. | In-Place |
| `tensor_scalar_mul` | List A, Number n | Multiplies every element by n. | In-Place |
| `tensor_add_scalar` | List A, Number n | Adds n to every element. | In-Place |

### 2. Element-Wise Math

Operations performed element-by-element (`A_ij op B_ij`). Dimensions must match.

| Function | Parameters | Description | Behavior |
| :--- | :--- | :--- | :--- |
| `tensor_add` | List A, List B | `A = A + B` | In-Place (Modifies A) |
| `tensor_sub` | List A, List B | `A = A - B` | In-Place (Modifies A) |
| `tensor_mul_elem` | List A, List B | `A = A * B` (Hadamard Product) | In-Place (Modifies A) |
| `tensor_div` | List A, List B | `A = A / B` (Safe division) | In-Place (Modifies A) |

### 3. Math Transformations

| Function | Parameters | Description | Behavior |
| :--- | :--- | :--- | :--- |
| `tensor_square` | List A | `x^2` for all elements. | In-Place |
| `tensor_sqrt` | List A | `sqrt(x)` for all elements. | In-Place |
| `tensor_exp` | List A | `e^x` for all elements. | In-Place |

### 4. Matrix Operations (Linear Algebra)

| Function | Parameters | Description | Return |
| :--- | :--- | :--- | :--- |
| `tensor_matmul` | List A, List B | Dot Product (`Rows A x Cols B`). | Returns New List |
| `tensor_transpose` | List A | Swaps rows and columns (`A^T`). | Returns New List |
| `tensor_sum` | List A, Axis | Sums elements. 1=Sum Rows, 0=Sum Cols. | Returns New List |
| `tensor_mean` | List A | Calculates the arithmetic mean of all items. | Returns Number |
| `tensor_argmax` | List A | Returns index of max value for each row. | Returns New List |

### 5. Activation Functions

All activations modify the list In-Place.

| Function | Derivative Function | Formula |
| :--- | :--- | :--- |
| `tensor_sigmoid` | `tensor_sigmoid_prime` | `1 / (1 + e^-x)` |
| `tensor_tanh` | `tensor_tanh_prime` | `tanh(x)` |
| `tensor_relu` | `tensor_relu_prime` | `max(0, x)` |
| `tensor_softmax` | N/A | Stable Softmax (Exp-Normalize) |

## 6. Optimizers & Regularization

High-performance Fused Kernels.

### `tensor_update_sgd`
Standard Stochastic Gradient Descent.

```ring
tensor_update_sgd(List W, List Grad, Number LR)
```
*   **W**: Weights Matrix (Modified In-Place).
*   **Grad**: Gradients Matrix.
*   **LR**: Learning Rate.

### `tensor_update_adam`
Adaptive Moment Estimation (Adam). Updates `W`, `M`, and `V` in a single pass.

```ring
tensor_update_adam(List W, List G, List M, List V, LR, Beta1, Beta2, Eps, T)
```
*   **W**: Weights.
*   **G**: Gradients.
*   **M**: Momentum (1st moment).
*   **V**: Velocity (2nd moment).
*   **LR, Beta1, Beta2, Eps**: Hyperparameters.
*   **T**: Current Time step (Integer).

### `tensor_dropout`
Randomly zeroes out elements.

```ring
tensor_dropout(List A, Number Rate)
```
*   **Rate**: Probability to drop (e.g., 0.2). Scales remaining elements automatically.

## 💻 Usage Example

```ring
loadlib("ring_tensor.dll") # or .so

# 1. Create Data
Matrix = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]

# 2. Math Operations
tensor_add_scalar(Matrix, 10.0)  # Add 10 to all
tensor_square(Matrix)            # Square all

# 3. Matrix Multiplication
Vec = [ [1.0], [1.0], [1.0] ]
Result = tensor_matmul(Matrix, Vec)

# 4. Activation
tensor_sigmoid(Result)

see Result
```

## ⚠️ Notes

*   **Lists as Pointers**: Ring passes lists by reference to C extensions. Functions marked "In-Place" will modify the original list variable in Ring.
*   **Double Precision**: Ensure all inputs are Numbers/Doubles. Strings inside lists will cause errors.
*   **Memory**: `tensor_matmul` and `tensor_transpose` allocate new memory. Ensure to handle large tensors carefully in loops.
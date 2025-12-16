/*
** RingTensor Extension
** Description: High-Performance Memory-Resident Tensor for RingML
** Header File
*/

#ifndef RING_TENSOR_H
#define RING_TENSOR_H

#include "ring.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h> 

#ifdef _WIN32
#define RING_EXPORT __declspec(dllexport)
#else
#define RING_EXPORT extern
#endif

/* --- The Tensor Structure (Lives in C Memory) --- */
typedef struct {
    double *data;   // Raw Double Array (Fastest access)
    int rows;
    int cols;
    int size;       // rows * cols
} tensor_t;

/* Tag to identify our pointers in Ring */
#define RING_VM_POINTER_TENSOR "tensor_t"

/* --- Prototypes --- */

/* Lifecycle */
RING_FUNC(ring_tensor_init);    // Malloc
RING_FUNC(ring_tensor_set);     // Set Value (for debugging/loading)
RING_FUNC(ring_tensor_get);     // Get Value (for debugging/saving)

/* Math (In-Place) */
RING_FUNC(ring_tensor_add);
RING_FUNC(ring_tensor_sub);
RING_FUNC(ring_tensor_mul_elem);
RING_FUNC(ring_tensor_div);
RING_FUNC(ring_tensor_scalar_mul);
RING_FUNC(ring_tensor_add_scalar);

/* Transformations (In-Place) */
RING_FUNC(ring_tensor_square);
RING_FUNC(ring_tensor_sqrt);
RING_FUNC(ring_tensor_exp);
RING_FUNC(ring_tensor_fill);
RING_FUNC(ring_tensor_random);

/* Matrix Ops */
RING_FUNC(ring_tensor_matmul);  // A * B -> C
RING_FUNC(ring_tensor_transpose);
RING_FUNC(ring_tensor_sum);     // Reduce to vector
RING_FUNC(ring_tensor_mean);

/* Activations (In-Place) */
RING_FUNC(ring_tensor_sigmoid);
RING_FUNC(ring_tensor_sigmoid_prime);
RING_FUNC(ring_tensor_tanh);
RING_FUNC(ring_tensor_tanh_prime);
RING_FUNC(ring_tensor_relu);
RING_FUNC(ring_tensor_relu_prime);
RING_FUNC(ring_tensor_softmax);

/* Optimizers (Fused Kernels) */
RING_FUNC(ring_tensor_update_sgd);
RING_FUNC(ring_tensor_update_adam);
RING_FUNC(ring_tensor_dropout);

/* Utilities */
RING_FUNC(ring_tensor_argmax);

/* Init */
RING_EXPORT void ringlib_init(RingState *pRingState);

#endif
/*
** RingTensor Extension - Header
** Updated for N-Dimensional Support (3D/4D)
*/

#ifndef RING_TENSOR_H
#define RING_TENSOR_H

#include "ring.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h> 

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#define RING_EXPORT __declspec(dllexport)
#else
#define RING_EXPORT extern
#endif

/* 
** The Tensor Structure (N-Dimensional Capable)
** We allow up to 4 Dimensions: [Batch, Heads, Rows, Cols]
** For 2D Tensors: Batch=1, Heads=1
*/
typedef struct {
    double *data;
    
    // Physical Dimensions
    int shape[4]; // [0]=Batch, [1]=Heads, [2]=Rows, [3]=Cols
    int ndim;     // Current active dimensions (2, 3, or 4)
    
    // Quick Aliases for Legacy Code (Pointing to shape[2] and shape[3])
    int rows; 
    int cols;
    
    int size;     // Total Elements
} tensor_t;

#define RING_VM_POINTER_TENSOR "tensor_t"

/* Library Entry Point */
RING_EXPORT void ringlib_init(RingState *pRingState);

/* --- PROTOTYPES --- */

/* Lifecycle */
RING_FUNC(ring_tensor_init);
RING_FUNC(ring_tensor_reshape); // <--- NEW: Change dimensions logically
RING_FUNC(ring_tensor_copy);

RING_FUNC(ring_tensor_get_data_ptr);

/* Batch Operations */
RING_FUNC(ring_tensor_matmul_batch); // <--- NEW: 3D MatMul (Batch x N x M) * (Batch x M x P)

RING_FUNC(ring_tensor_set);     // Set single value
RING_FUNC(ring_tensor_get);     // Get single value

/* Properties */
RING_FUNC(ring_tensor_get_rows);
RING_FUNC(ring_tensor_get_cols);

/* 2. Element-Wise Math (In-Place / Parallelized) */
RING_FUNC(ring_tensor_add);
RING_FUNC(ring_tensor_sub);
RING_FUNC(ring_tensor_mul_elem);
RING_FUNC(ring_tensor_div);
RING_FUNC(ring_tensor_scalar_mul);
RING_FUNC(ring_tensor_add_scalar);
RING_FUNC(ring_tensor_sub_scalar); // New

/* 3. Transformations & Activations (In-Place / Parallelized) */
RING_FUNC(ring_tensor_fill);
RING_FUNC(ring_tensor_random);
RING_FUNC(ring_tensor_square);
RING_FUNC(ring_tensor_sqrt);
RING_FUNC(ring_tensor_exp);
RING_FUNC(ring_tensor_sigmoid);
RING_FUNC(ring_tensor_sigmoid_prime);
RING_FUNC(ring_tensor_tanh);
RING_FUNC(ring_tensor_tanh_prime);
RING_FUNC(ring_tensor_relu);
RING_FUNC(ring_tensor_relu_prime);
RING_FUNC(ring_tensor_softmax);

/* 4. Matrix Operations (Heavily Optimized) */
RING_FUNC(ring_tensor_matmul);      // A * B
RING_FUNC(ring_tensor_transpose);   // A.T
RING_FUNC(ring_tensor_sum);         // Axis 0 or 1
RING_FUNC(ring_tensor_mean);
RING_FUNC(ring_tensor_argmax);
RING_FUNC(ring_tensor_add_row_vec); // Broadcasting (Bias add)

/* 5. NLP & Transformer Kernels (Advanced) */
RING_FUNC(ring_tensor_embedding_forward);
RING_FUNC(ring_tensor_embedding_backward);

RING_FUNC(ring_tensor_layernorm);

RING_FUNC(ring_tensor_attention_fast);   // Standard Attention
RING_FUNC(ring_tensor_attention_causal); // GPT Masked Attention
RING_FUNC(ring_tensor_mha_backward_fast);
RING_FUNC(ring_tensor_attention_batch);

RING_FUNC(ring_tensor_select_columns);   // Fast Slicing
RING_FUNC(ring_tensor_insert_columns);   // Fast Concat

// --- NEW ---
RING_FUNC(ring_tensor_slice_rows);
RING_FUNC(ring_tensor_insert_rows);

/* 6. Optimizers (Fused Kernels) */
RING_FUNC(ring_tensor_update_sgd);
RING_FUNC(ring_tensor_update_adam);
RING_FUNC(ring_tensor_dropout);

RING_FUNC(ring_tensor_crossentropy_loss);
RING_FUNC(ring_tensor_crossentropy_backward);

/* 7. Utilities */
RING_FUNC(ring_tensor_get_cores);
RING_FUNC(ring_tensor_set_threads);


RING_FUNC(ring_tensor_set_from_list);
RING_FUNC(ring_tensor_set_one_hot);

/* Persistence (Binary & Quantized) */
RING_FUNC(ring_tensor_save);
RING_FUNC(ring_tensor_load);
RING_FUNC(ring_tensor_save_fp32);
RING_FUNC(ring_tensor_load_fp32);
RING_FUNC(ring_tensor_load_inplace);
RING_FUNC(ring_tensor_load_fp32_inplace);

RING_FUNC(ring_tensor_clip_global_norm);
RING_FUNC(ring_tensor_clip_tensor);

// --- NEW ---
RING_FUNC(ring_tensor_gelu);
RING_FUNC(ring_tensor_gelu_prime);

// --- NEW ---
RING_FUNC(ring_tensor_sum_squares);
RING_FUNC(ring_tensor_repeat_rows);
RING_FUNC(ring_tensor_from_memory);

// --- NEW ---
RING_FUNC(ring_tensor_attention_linear_causal);
RING_FUNC(ring_tensor_attention_linear_optimized);
RING_FUNC(ring_tensor_attention_linear_backward);
RING_FUNC(ring_tensor_attention_linear_global_backward);

// --- NEW ---
RING_FUNC(ring_tensor_attention_multihead);
RING_FUNC(ring_tensor_attention_multihead_backward);

RING_FUNC(ring_tensor_set_one_hot_ptr);

#endif
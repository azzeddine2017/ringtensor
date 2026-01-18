/*
** RingTensor Extension Implementation
** Optimized for Dual-Core / Hyper-threaded CPUs
** Fixed for MSVC C3015 Error & High Overhead
*/

#include "ring_tensor.h"

/* 
** Cache Block Size
** 32 doubles * 8 bytes = 256 bytes (Fits easily in L1 Cache) 
*/
#define TILE_SIZE 32

/* 
** TUNING:
** Threshold raised to 10,000 to prevent overhead on small matrices.
** On i3-5005U, small tasks are faster in serial mode.
*/
#define PARALLEL_THRESHOLD 50000

/* --- Memory Management --- */
void ring_tensor_free(void *pState, void *pPointer) {
    tensor_t *t = (tensor_t *)pPointer;
    if (t != NULL) {
        if (t->data != NULL) free(t->data);
        free(t);
    }
}

/*
** Destructor for Borrowed Memory
** Frees the struct ONLY, not the data (owned by AlQalam/C++)
*/
void ring_tensor_free_struct_only(void *pState, void *pPointer) {
    tensor_t *t = (tensor_t *)pPointer;
    if (t != NULL) {
        // DO NOT FREE t->data!
        free(t);
    }
}

/* ==================================================================== */
/* --- 1. LIFECYCLE --------------------------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_init) {
    int rows, cols;
    tensor_t *t;

    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    rows = (int)RING_API_GETNUMBER(1);
    cols = (int)RING_API_GETNUMBER(2);
    
    if (rows <= 0 || cols <= 0) { RING_API_ERROR("Invalid Dims"); return; }

    t = (tensor_t *)malloc(sizeof(tensor_t));
    if(!t) { RING_API_ERROR("Malloc Fail"); return; }
    
    // Default to 2D
    t->ndim = 2;
    t->shape[0] = 1; // Batch
    t->shape[1] = 1; // Heads
    t->shape[2] = rows;
    t->shape[3] = cols;
    
    // Legacy Aliases
    t->rows = rows;
    t->cols = cols;
    
    t->size = rows * cols;
    
    t->data = (double *)calloc(t->size, sizeof(double));
    if (!t->data) { free(t); RING_API_ERROR("Malloc Data Fail"); return; }
    
    RING_API_RETMANAGEDCPOINTER(t, RING_VM_POINTER_TENSOR, ring_tensor_free);
}

/*
** Reshape Tensor
** Usage: tensor_reshape(pTensor, Batch, Heads, Rows, Cols)
** Pass 1 for unused dimensions. Product must equal total size.
*/
RING_FUNC(ring_tensor_reshape) {
    tensor_t *t;
    int b, h, r, c;
    int new_size;

    if (RING_API_PARACOUNT != 5) {
        RING_API_ERROR("Reshape requires 4 dims (Batch, Heads, Rows, Cols). Use 1 for unused.");
        return;
    }

    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    b = (int)RING_API_GETNUMBER(2);
    h = (int)RING_API_GETNUMBER(3);
    r = (int)RING_API_GETNUMBER(4);
    c = (int)RING_API_GETNUMBER(5);
    
    new_size = b * h * r * c;
    
    if (new_size != t->size) {
        RING_API_ERROR("Reshape Error: Total size cannot change.");
        return;
    }
    
    // Update Logical Shape
    t->shape[0] = b;
    t->shape[1] = h;
    t->shape[2] = r;
    t->shape[3] = c;
    
    // Determine ndim for internal logic
    if (b > 1) t->ndim = (h > 1) ? 4 : 3;
    else t->ndim = 2;
    
    // Update legacy aliases (Always points to last two dims)
    t->rows = r;
    t->cols = c;
}

/*
** Copy Tensor (Deep Copy)
** Creates a new independent tensor with the same data and shape.
** Optimization: Uses memcpy for maximum throughput.
*/
RING_FUNC(ring_tensor_copy) {
    tensor_t *Src, *Dest;
    size_t bytes;
    int i;

    if (RING_API_PARACOUNT != 1) {
        RING_API_ERROR(RING_API_MISS1PARA);
        return;
    }

    Src = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);

    // 1. Allocate Struct
    Dest = (tensor_t *)malloc(sizeof(tensor_t));
    if (!Dest) {
        RING_API_ERROR("Malloc Failed (Struct)");
        return;
    }

    // 2. Copy Metadata (Dimensions)
    Dest->rows = Src->rows;
    Dest->cols = Src->cols;
    Dest->size = Src->size;
    Dest->ndim = Src->ndim;
    
    for(i=0; i<4; i++) Dest->shape[i] = Src->shape[i];

    // 3. Allocate Data
    Dest->data = (double *)malloc(Dest->size * sizeof(double));
    if (!Dest->data) {
        free(Dest);
        RING_API_ERROR("Malloc Failed (Data)");
        return;
    }

    // 4. Copy Data (Memcpy is highly optimized by CPU)
    bytes = Dest->size * sizeof(double);
    memcpy(Dest->data, Src->data, bytes);

    // 5. Return Managed Pointer
    RING_API_RETMANAGEDCPOINTER(Dest, RING_VM_POINTER_TENSOR, ring_tensor_free);
}

/*
** Batch Matrix Multiplication (3D)
** A: (Batch, RowsA, ColsA)
** B: (Batch, RowsB, ColsB) -> ColsA must equal RowsB
** C: (Batch, RowsA, ColsB)
*/
RING_FUNC(ring_tensor_matmul_batch) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *C = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int batch = A->shape[0];
    int rA = A->rows; int cA = A->cols; int cB = B->cols;
    int strideA = rA * cA; int strideB = cA * cB; int strideC = rA * cB;
    
    int b_idx, i, j, k, ii, jj, kk;
    int i_max, j_max, k_max;
    double *ptrA, *ptrB, *ptrC, *rowC, *rowA;

    if (B->shape[0] != batch) { RING_API_ERROR("BMM Mismatch"); return; }

    // Parallelize Batches
    #pragma omp parallel for private(b_idx, ptrA, ptrB, ptrC, i, j, k, ii, jj, kk, i_max, j_max, k_max, rowC, rowA)
    for (b_idx = 0; b_idx < batch; b_idx++) {
        
        ptrA = &A->data[b_idx * strideA];
        ptrB = &B->data[b_idx * strideB];
        ptrC = &C->data[b_idx * strideC];
        
        // Zero out Batch C
        memset(ptrC, 0, strideC * sizeof(double));
        
        // Tiled MatMul for this Batch
        for (ii = 0; ii < rA; ii += TILE_SIZE) {
            i_max = (ii + TILE_SIZE > rA) ? rA : ii + TILE_SIZE;

            for (kk = 0; kk < cA; kk += TILE_SIZE) {
                k_max = (kk + TILE_SIZE > cA) ? cA : kk + TILE_SIZE;

                for (jj = 0; jj < cB; jj += TILE_SIZE) {
                    j_max = (jj + TILE_SIZE > cB) ? cB : jj + TILE_SIZE;

                    for (i = ii; i < i_max; i++) {
                        rowC = &ptrC[i * cB];
                        rowA = &ptrA[i * cA];
                        
                        for (k = kk; k < k_max; k++) {
                            double valA = rowA[k];
                            if (valA == 0.0) continue;
                            
                            double *rowB = &ptrB[k * cB];
                            for (j = jj; j < j_max; j++) {
                                rowC[j] += valA * rowB[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

RING_FUNC(ring_tensor_set) {
    tensor_t *t;
    int r, c;
    double val;

    if (RING_API_PARACOUNT != 4) return;
    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    r = (int)RING_API_GETNUMBER(2);
    c = (int)RING_API_GETNUMBER(3);
    val = RING_API_GETNUMBER(4);
    
    // Bounds check
    if (r < 1 || r > t->rows || c < 1 || c > t->cols) return;
    t->data[(r-1) * t->cols + (c-1)] = val;
}

RING_FUNC(ring_tensor_get) {
    tensor_t *t;
    int r, c;

    if (RING_API_PARACOUNT != 3) return;
    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    r = (int)RING_API_GETNUMBER(2);
    c = (int)RING_API_GETNUMBER(3);
    
    if (r < 1 || r > t->rows || c < 1 || c > t->cols) {
        RING_API_RETNUMBER(0.0);
        return;
    }
    RING_API_RETNUMBER(t->data[(r-1) * t->cols + (c-1)]);
}

/* ==================================================================== */
/* --- 2. ELEMENT-WISE MATH (OPTIMIZED) ------------------------------- */
/* ==================================================================== */

void tensor_op_elem(void *pPointer, int op) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int i;
    int size = A->size;

    if (A->size != B->size) { RING_API_ERROR("Tensor Size Mismatch"); return; }

    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) {
        switch(op) {
            case 1: A->data[i] += B->data[i]; break;
            case 2: A->data[i] -= B->data[i]; break;
            case 3: A->data[i] *= B->data[i]; break;
            case 4: A->data[i] = (B->data[i] != 0) ? A->data[i] / B->data[i] : 0.0; break;
        }
    }
}

RING_FUNC(ring_tensor_add)      { tensor_op_elem(pPointer, 1); }
RING_FUNC(ring_tensor_sub)      { tensor_op_elem(pPointer, 2); }
RING_FUNC(ring_tensor_mul_elem) { tensor_op_elem(pPointer, 3); }
RING_FUNC(ring_tensor_div)      { tensor_op_elem(pPointer, 4); }

RING_FUNC(ring_tensor_scalar_mul) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i; 
    int size = A->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) A->data[i] *= v;
}

RING_FUNC(ring_tensor_add_scalar) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i; 
    int size = A->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) A->data[i] += v;
}

RING_FUNC(ring_tensor_sub_scalar) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i;
    int size = A->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) A->data[i] -= v;
}

/* ==================================================================== */
/* --- 3. TRANSFORMS & ACTIVATIONS ------------------------------------ */
/* ==================================================================== */

RING_FUNC(ring_tensor_fill) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i;
    int size = A->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) A->data[i] = v;
}

RING_FUNC(ring_tensor_random) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    // Keep Random Serial for Reproducibility
    for(i=0; i<A->size; i++) A->data[i] = (double)rand() / (double)RAND_MAX;
}

void tensor_act(void *pPointer, int op) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    int size = A->size;
    double v;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD) private(v)
    for(i=0; i<size; i++) {
        v = A->data[i];
        switch(op) {
            case 1: A->data[i] = v * v; break; // Square
            case 2: A->data[i] = sqrt(v); break; // Sqrt
            case 3: A->data[i] = exp(v); break; // Exp
            case 4: A->data[i] = 1.0 / (1.0 + exp(-v)); break; // Sigmoid
            case 5: A->data[i] = v * (1.0 - v); break; // SigmoidPrime
            case 6: A->data[i] = tanh(v); break; // Tanh
            case 7: A->data[i] = 1.0 - (v * v); break; // TanhPrime
            case 8: A->data[i] = (v > 0) ? v : 0; break; // ReLU
            case 9: A->data[i] = (v > 0) ? 1.0 : 0.0; break; // ReLUPrime
        }
    }
}

RING_FUNC(ring_tensor_square)        { tensor_act(pPointer, 1); }
RING_FUNC(ring_tensor_sqrt)          { tensor_act(pPointer, 2); }
RING_FUNC(ring_tensor_exp)           { tensor_act(pPointer, 3); }
RING_FUNC(ring_tensor_sigmoid)       { tensor_act(pPointer, 4); }
RING_FUNC(ring_tensor_sigmoid_prime) { tensor_act(pPointer, 5); }
RING_FUNC(ring_tensor_tanh)          { tensor_act(pPointer, 6); }
RING_FUNC(ring_tensor_tanh_prime)    { tensor_act(pPointer, 7); }
RING_FUNC(ring_tensor_relu)          { tensor_act(pPointer, 8); }
RING_FUNC(ring_tensor_relu_prime)    { tensor_act(pPointer, 9); }

RING_FUNC(ring_tensor_softmax) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int r, c;
    
    // Row-wise Softmax
    #pragma omp parallel for if(T->rows > 64) private(c)
    for(r=0; r<T->rows; r++) {
        double maxVal = -DBL_MAX;
        double sum = 0.0;
        int offset = r * T->cols;
        double invSum;
        
        for(c=0; c<T->cols; c++) if(T->data[offset+c] > maxVal) maxVal = T->data[offset+c];
        
        for(c=0; c<T->cols; c++) {
            T->data[offset+c] = exp(T->data[offset+c] - maxVal);
            sum += T->data[offset+c];
        }
        
        invSum = (sum != 0) ? 1.0/sum : 0.0;
        for(c=0; c<T->cols; c++) T->data[offset+c] *= invSum;
    }
}

/* ==================================================================== */
/* --- 4. MATRIX OPS (OPTIMIZED MATMUL) ------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_matmul) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *C = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int rA = A->rows; int cA = A->cols; int cB = B->cols;
    
    // تعريف المتغيرات في البداية (MSVC C89 Compliance)
    int i, j, k, ii, jj, kk;
    int i_max, j_max, k_max;
    double *rowC, *rowA;

    if (cA != B->rows) { RING_API_ERROR("MatMul Dims Mismatch"); return; }
    
    // تحسين 1: استخدام memset لتصفير المصفوفة (أسرع من حلقة for)
    memset(C->data, 0, (size_t)rA * cB * sizeof(double));

    // --- TILED ALGORITHM (Cache Friendly) ---
    // نوزع الكتل (Blocks) على الأنوية
    // private(...) ضرورية جداً هنا لمنع تداخل الخيوط
    
    #pragma omp parallel for schedule(static) private(ii, jj, kk, i, j, k, i_max, j_max, k_max, rowC, rowA)
    for (ii = 0; ii < rA; ii += TILE_SIZE) {
        
        i_max = (ii + TILE_SIZE > rA) ? rA : ii + TILE_SIZE;

        for (kk = 0; kk < cA; kk += TILE_SIZE) {
            k_max = (kk + TILE_SIZE > cA) ? cA : kk + TILE_SIZE;

            for (jj = 0; jj < cB; jj += TILE_SIZE) {
                j_max = (jj + TILE_SIZE > cB) ? cB : jj + TILE_SIZE;

                // Inner loops: الضرب الفعلي داخل البلاطة (Tile)
                // هذه الحلقات صغيرة وتناسب الكاش
                for (i = ii; i < i_max; i++) {
                    rowC = &C->data[i * cB];
                    rowA = &A->data[i * cA];
                    
                    for (k = kk; k < k_max; k++) {
                        double valA = rowA[k];
                        
                        // Sparse Optimization: تخطي الأصفار
                        if (valA == 0.0) continue;

                        double *rowB = &B->data[k * cB];
                        
                        // Vectorized loop (Compiler will use SIMD/AVX here)
                        for (j = jj; j < j_max; j++) {
                            rowC[j] += valA * rowB[j];
                        }
                    }
                }
            }
        }
    }
}

RING_FUNC(ring_tensor_transpose) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *C = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    
    int rA = A->rows; 
    int cA = A->cols;
    
    // تعريف المتغيرات في البداية (توافق MSVC)
    int i, j, ii, jj;
    int i_max, j_max;
    
    // --- TILED TRANSPOSE ALGORITHM ---
    // نقوم بتوزيع كتل الصفوف (Rows Blocks) على الأنوية
    
    #pragma omp parallel for schedule(static) private(ii, jj, i, j, i_max, j_max)
    for (ii = 0; ii < rA; ii += TILE_SIZE) {
        
        i_max = (ii + TILE_SIZE > rA) ? rA : ii + TILE_SIZE;
        
        for (jj = 0; jj < cA; jj += TILE_SIZE) {
            
            j_max = (jj + TILE_SIZE > cA) ? cA : jj + TILE_SIZE;
            
            // --- OPTIMIZATION: Loop Order Swapped ---
            // جعلنا الحلقة الخارجية j والداخلية i
            // الهدف: الكتابة في C تكون متسلسلة (Sequential Write)
            // C[... + i] أفضل بكثير من C[... + j*stride]
            
            for (j = jj; j < j_max; j++) {
                for (i = ii; i < i_max; i++) {
                    // C[j][i] = A[i][j]
                    // الآن C يُكتب فيه بشكل متتابع، و A يُقرأ منه بقفزات
                    // القراءة العشوائية أهون على المعالج من الكتابة العشوائية
                    C->data[j * rA + i] = A->data[i * cA + j];
                }
            }
        }
    }
}

RING_FUNC(ring_tensor_sum) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int axis = (int)RING_API_GETNUMBER(2);
    tensor_t *R = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    // تعريف المتغيرات في البداية (توافق C89/MSVC)
    int r, c;
    double s;
    double *ptr, *src, *dst, *rowPtr;

    // تصفير النتيجة
    memset(R->data, 0, R->size * sizeof(double));

    if (axis == 1) { // Sum Rows (Collapse Cols) -> Result is Col Vector
        
        // المتغيرات s, c, ptr يجب أن تكون private لكل خيط
        #pragma omp parallel for if(T->rows > 64) private(c, s, ptr)
        for(r=0; r<T->rows; r++) {
            s = 0;
            ptr = &T->data[r * T->cols];
            
            for(c=0; c<T->cols; c++) {
                s += ptr[c];
            }
            R->data[r] = s;
        }

    } else { // Sum Cols (Collapse Rows) -> Result is Row Vector (Bias Gradient)
        
        /* 
        ** نبقيها تسلسلية (Serial) للأمان.
        ** توازي هذا الجزء يتطلب Atomic Operations أو Reduction Array
        ** وهو مكلف أكثر من الفائدة في أحجام الباتش الصغيرة/المتوسطة.
        */
        src = T->data;
        dst = R->data;

        for(r=0; r<T->rows; r++) {
            rowPtr = &src[r * T->cols];
            
            for(c=0; c<T->cols; c++) {
                dst[c] += rowPtr[c];
            }
        }
    }
}

RING_FUNC(ring_tensor_add_row_vec) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    
    // تعريف المتغيرات في البداية (توافق C89/ANSI C)
    int i, j;
    double *rowA, *rowB;
    
    if (A->cols != B->cols) { RING_API_ERROR("Dim Mismatch"); return; }

    // 1. تحديد المتغيرات الخاصة بكل خيط (Thread) لضمان عدم تداخل الذاكرة
    // rowA, rowB, j يجب أن تكون خاصة لكل عملية
    #pragma omp parallel for if(A->rows > 32) private(j, rowA, rowB)
    for(i=0; i<A->rows; i++) {
        
        // حساب المؤشرات
        rowA = &A->data[i * A->cols];
        rowB = B->data; // هذا ثابت، لكن تعيينه هنا آمن وسريع
        
        // الحلقة الداخلية (Vectorized)
        for(j=0; j<A->cols; j++) {
            rowA[j] += rowB[j];
        }
    }
}

RING_FUNC(ring_tensor_mean) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double sum = 0;
    int i;
    #pragma omp parallel for reduction(+:sum) if(T->size > PARALLEL_THRESHOLD)
    for(i=0; i<T->size; i++) sum += T->data[i];
    RING_API_RETNUMBER(sum / T->size);
}

RING_FUNC(ring_tensor_argmax) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *R = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int r, c;
    
    #pragma omp parallel for if(T->rows > 64) private(c)
    for(r=0; r<T->rows; r++) {
        double maxVal = -DBL_MAX;
        int maxIdx = 1;
        int offset = r * T->cols;
        for(c=0; c<T->cols; c++) {
            if(T->data[offset+c] > maxVal) {
                maxVal = T->data[offset+c];
                maxIdx = c + 1;
            }
        }
        R->data[r] = (double)maxIdx;
    }
}

/*
** Slice Rows (Extraction)
** Copies 'count' rows starting from 'start_row' in Src to Dest.
** Optimization: Uses single memcpy because rows are contiguous in memory.
*/
RING_FUNC(ring_tensor_slice_rows) {
    tensor_t *Src, *Dest;
    int start_row, count;
    
    if (RING_API_PARACOUNT != 4) {
        RING_API_ERROR(RING_API_MISS4PARA);
        return;
    }

    Src = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    Dest = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    start_row = (int)RING_API_GETNUMBER(3); // 1-based index
    count = (int)RING_API_GETNUMBER(4);     // Number of rows

    // 1. Validation
    if (start_row < 1 || start_row + count - 1 > Src->rows) {
        RING_API_ERROR("Slice Rows: Index out of bounds");
        return;
    }
    if (Dest->cols != Src->cols) {
        RING_API_ERROR("Slice Rows: Column count mismatch");
        return;
    }
    if (Dest->rows != count) {
        RING_API_ERROR("Slice Rows: Destination rows must match count");
        return;
    }

    // 2. Calculation (The Fast Part)
    // Calculate where to start reading in Source (0-based index)
    // Offset = (StartRow - 1) * NumberOfColumns
    int src_offset_idx = (start_row - 1) * Src->cols;
    
    // Calculate total bytes to copy
    // Bytes = NumberOfRowsToCopy * NumberOfColumns * SizeOfDouble
    size_t total_elements = (size_t)count * Src->cols;
    size_t bytes = total_elements * sizeof(double);
    
    // 3. Execution (Single Memcpy)
    // Copy directly from Src memory address to Dest memory address
    memcpy(Dest->data, &Src->data[src_offset_idx], bytes);
}

/*
** Insert Rows (Injection)
** Copies all rows from Src into Dest starting at 'start_row'.
*/
RING_FUNC(ring_tensor_insert_rows) {
    tensor_t *Dest, *Src;
    int start_row;

    if (RING_API_PARACOUNT != 3) {
        RING_API_ERROR(RING_API_MISS3PARA);
        return;
    }

    Dest = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    Src = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    start_row = (int)RING_API_GETNUMBER(3); // 1-based index

    // Validation
    if (start_row < 1 || start_row + Src->rows - 1 > Dest->rows) {
        RING_API_ERROR("Insert Rows: Index out of bounds or Src too big");
        return;
    }
    if (Dest->cols != Src->cols) {
        RING_API_ERROR("Insert Rows: Column count mismatch");
        return;
    }

    // Math
    size_t offset_elems = (size_t)(start_row - 1) * Dest->cols;
    size_t copy_elems   = (size_t)Src->rows * Src->cols;
    size_t copy_bytes   = copy_elems * sizeof(double);

    // Blazing fast copy
    memcpy(&Dest->data[offset_elems], Src->data, copy_bytes);
}

/* ==================================================================== */
/* --- 5. NLP & TRANSFORMER KERNELS (OpenMP Optimized) ---------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_embedding_forward) {
    tensor_t *Ind = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *W   = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int total = Ind->size;
    int dim = W->cols;
    int vocab = W->rows;
    int i;
    
    #pragma omp parallel for if(total > 32)
    for(i=0; i<total; i++) {
        int idx = (int)Ind->data[i];
        
        // تحويل من Ring indexing (1-based) إلى C indexing (0-based)
        idx = idx - 1;
        
        // Bounds checking
        if (idx < 0) idx = 0; 
        if (idx >= vocab) idx = vocab - 1;
        
        memcpy(&Out->data[i * dim], &W->data[idx * dim], dim * sizeof(double));
    }
}

RING_FUNC(ring_tensor_embedding_backward) {
    tensor_t *Ind = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *GOut = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *GW = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int total = Ind->size;
    int dim = GW->cols;
    int i, j;
    
    for(i=0; i<total; i++) {
        int idx = (int)Ind->data[i];
        
        // ✅ تحويل من Ring indexing (1-based) إلى C indexing (0-based)
        idx = idx - 1;
        
        // Bounds checking
        if (idx < 0 || idx >= GW->rows) continue;
        
        double *g_src = &GOut->data[i * dim];
        double *g_dst = &GW->data[idx * dim];
        
        for(j=0; j<dim; j++) g_dst[j] += g_src[j];
    }
}

RING_FUNC(ring_tensor_layernorm) {
    tensor_t *X = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *G = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Y = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    double eps = RING_API_GETNUMBER(5);
    int r, c, rows = X->rows, cols = X->cols;
    
    #pragma omp parallel for if(rows > 32) private(c)
    for(r=0; r<rows; r++) {
        double mean = 0, var = 0;
        double *px = &X->data[r*cols];
        double *py = &Y->data[r*cols];
        double invStd;
        
        for(c=0; c<cols; c++) mean += px[c];
        mean /= cols;
        
        for(c=0; c<cols; c++) var += (px[c]-mean)*(px[c]-mean);
        var /= cols;
        
        invStd = 1.0 / sqrt(var + eps);
        for(c=0; c<cols; c++) {
            py[c] = ((px[c] - mean) * invStd * G->data[c]) + B->data[c];
        }
    }
}

RING_FUNC(ring_tensor_attention_fast) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    double scale = RING_API_GETNUMBER(5);
    
    int seq = Q->rows;
    int dim = Q->cols;
    int i, j, k;
    
    // Allocating temp memory inside parallel region is expensive.
    // Optimization: Do parallel loop but handle malloc carefully or use small stack buffer if dim is small.
    // For large sequence, use malloc.
    
    #pragma omp parallel private(i, j, k)
    {
        double *scores = (double *)malloc(seq * sizeof(double));
        if(scores) {
            #pragma omp for
            for(i=0; i<seq; i++) {
                double *q_row = &Q->data[i*dim];
                double *out_row = &Out->data[i*dim];
                double maxVal = -DBL_MAX;
                double sum = 0;
                double invSum;
                
                // QK^T
                for(j=0; j<seq; j++) {
                    double *k_row = &K->data[j*dim];
                    double dot = 0;
                    for(k=0; k<dim; k++) dot += q_row[k] * k_row[k];
                    scores[j] = dot * scale;
                }
                
                // Softmax
                for(j=0; j<seq; j++) if(scores[j] > maxVal) maxVal = scores[j];
                for(j=0; j<seq; j++) {
                    scores[j] = exp(scores[j] - maxVal);
                    sum += scores[j];
                }
                invSum = 1.0/sum;
                for(j=0; j<seq; j++) scores[j] *= invSum;
                
                // Score * V
                memset(out_row, 0, dim * sizeof(double));
                for(j=0; j<seq; j++) {
                    double s = scores[j];
                    double *v_row = &V->data[j*dim];
                    for(k=0; k<dim; k++) out_row[k] += s * v_row[k];
                }
            }
            free(scores);
        }
    }
}

RING_FUNC(ring_tensor_attention_causal) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    double scale = RING_API_GETNUMBER(5);
    
    int seq = Q->rows;
    int dim = Q->cols;
    int i, j, k;
    
    #pragma omp parallel private(i, j, k)
    {
        double *scores = (double *)malloc(seq * sizeof(double));
        if(scores) {
            #pragma omp for
            for(i=0; i<seq; i++) {
                double *q_row = &Q->data[i*dim];
                double *out_row = &Out->data[i*dim];
                double maxVal = -1e9;
                double sum = 0;
                double invSum;
                
                // Masked QK^T
                for(j=0; j<seq; j++) {
                    if (j > i) { scores[j] = -1e9; continue; }
                    
                    double *k_row = &K->data[j*dim];
                    double dot = 0;
                    for(k=0; k<dim; k++) dot += q_row[k] * k_row[k];
                    scores[j] = dot * scale;
                }
                
                // Softmax
                for(j=0; j<seq; j++) if(scores[j] > maxVal) maxVal = scores[j];
                for(j=0; j<seq; j++) {
                    scores[j] = exp(scores[j] - maxVal);
                    sum += scores[j];
                }
                invSum = 1.0/sum;
                for(j=0; j<seq; j++) scores[j] *= invSum;
                
                // Score * V
                memset(out_row, 0, dim * sizeof(double));
                for(j=0; j<seq; j++) {
                    double s = scores[j];
                    if(s < 1e-9) continue;
                    double *v_row = &V->data[j*dim];
                    for(k=0; k<dim; k++) out_row[k] += s * v_row[k];
                }
            }
            free(scores);
        }
    }
}

RING_FUNC(ring_tensor_select_columns) {
    tensor_t *Src = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *Dest = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int start = (int)RING_API_GETNUMBER(3);
    int count = (int)RING_API_GETNUMBER(4);
    int r;
    int src_off;
    size_t bytes;
    
    if (start < 1 || start + count - 1 > Src->cols) { RING_API_ERROR("Bounds"); return; }
    
    src_off = start - 1;
    bytes = count * sizeof(double);
    
    #pragma omp parallel for if(Src->rows > 64)
    for(r=0; r<Src->rows; r++) {
        memcpy(&Dest->data[r*Dest->cols], &Src->data[r*Src->cols + src_off], bytes);
    }
}

RING_FUNC(ring_tensor_insert_columns) {
    tensor_t *Dest = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *Src = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    int start = (int)RING_API_GETNUMBER(3);
    int r;
    int dest_off;
    size_t bytes;
    
    if (start < 1 || start + Src->cols - 1 > Dest->cols) { RING_API_ERROR("Bounds"); return; }
    
    dest_off = start - 1;
    bytes = Src->cols * sizeof(double);
    
    #pragma omp parallel for if(Dest->rows > 64)
    for(r=0; r<Dest->rows; r++) {
        memcpy(&Dest->data[r*Dest->cols + dest_off], &Src->data[r*Src->cols], bytes);
    }
}

/*
** Fused Batch Attention (Fast & Causal)
** Handles [Batch, Seq, Dim] in one go using OpenMP.
** 
** Params: 
** 1. Q, 2. K, 3. V, 4. Out (All Flattened: Batch*Seq*Dim)
** 5. Scale, 6. BatchSize, 7. SeqLen, 8. Dim, 9. IsCausal (1=Yes, 0=No)
*/
RING_FUNC(ring_tensor_attention_batch) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    
    double scale  = RING_API_GETNUMBER(5);
    int batch     = (int)RING_API_GETNUMBER(6);
    int seq       = (int)RING_API_GETNUMBER(7);
    int dim       = (int)RING_API_GETNUMBER(8);
    int is_causal = (int)RING_API_GETNUMBER(9);
    
    int b, i, j, k;
    int stride = seq * dim; // Size of one sentence block
    
    // Parallelize over Batches (The most efficient way)
    #pragma omp parallel for private(b, i, j, k)
    for (b = 0; b < batch; b++) {
        
        // Pointers to the start of the current batch
        double *q_base = &Q->data[b * stride];
        double *k_base = &K->data[b * stride];
        double *v_base = &V->data[b * stride];
        double *o_base = &Out->data[b * stride];
        
        // Temp scores buffer (per thread/batch)
        double *scores = (double *)malloc(seq * sizeof(double));
        
        if (scores) {
            // Loop over Sequence (Rows)
            for(i = 0; i < seq; i++) {
                double *q_row = &q_base[i * dim];
                double *o_row = &o_base[i * dim];
                
                // 1. Q . K^T
                for(j = 0; j < seq; j++) {
                    // Causal Masking Logic
                    if (is_causal && j > i) {
                        scores[j] = -1e9;
                        continue;
                    }
                    
                    double *k_row = &k_base[j * dim];
                    double dot = 0.0;
                    for(k = 0; k < dim; k++) {
                        dot += q_row[k] * k_row[k];
                    }
                    scores[j] = dot * scale;
                }
                
                // 2. Softmax
                double maxVal = -1e9;
                for(j=0; j<seq; j++) if(scores[j] > maxVal) maxVal = scores[j];
                
                double sum = 0.0;
                for(j=0; j<seq; j++) {
                    scores[j] = exp(scores[j] - maxVal);
                    sum += scores[j];
                }
                
                double invSum = 1.0 / sum;
                for(j=0; j<seq; j++) scores[j] *= invSum;
                
                // 3. Scores . V
                memset(o_row, 0, dim * sizeof(double));
                for(j=0; j<seq; j++) {
                    double s = scores[j];
                    if (s < 1e-9) continue; // Optimization
                    
                    double *v_row = &v_base[j * dim];
                    for(k=0; k<dim; k++) {
                        o_row[k] += s * v_row[k];
                    }
                }
            }
            free(scores);
        }
    }
}

/* ==================================================================== */
/* --- 6. OPTIMIZERS -------------------------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_update_adam) {
    tensor_t *W = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *G = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *M = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    
    double lr = RING_API_GETNUMBER(5);
    double b1 = RING_API_GETNUMBER(6);
    double b2 = RING_API_GETNUMBER(7);
    double eps= RING_API_GETNUMBER(8);
    int t = (int)RING_API_GETNUMBER(9);
    double weight_decay = RING_API_GETNUMBER(10);
    
    double corr1 = 1.0 - pow(b1, t);
    double corr2 = 1.0 - pow(b2, t);
    if(corr1 < 1e-9) corr1 = 1e-9;
    if(corr2 < 1e-9) corr2 = 1e-9;
    
    int i;
    int size = W->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) {
        double g = G->data[i];
        
        // تحديث momentum
        M->data[i] = (b1 * M->data[i]) + ((1.0 - b1) * g);
        V->data[i] = (b2 * V->data[i]) + ((1.0 - b2) * g * g);
        
        double m_hat = M->data[i] / corr1;
        double v_hat = V->data[i] / corr2;
        if(v_hat < 0) v_hat = 0;
        
        // تحديث الأوزان مع Adam
        W->data[i] -= (lr * m_hat) / (sqrt(v_hat) + eps);
        
        // تطبيق Weight Decay مباشرة (AdamW)
        W->data[i] -= lr * weight_decay * W->data[i];
    }
}

RING_FUNC(ring_tensor_update_sgd) {
    tensor_t *W = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *G = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    double lr = RING_API_GETNUMBER(3);
    int i;
    int size = W->size;
    
    #pragma omp parallel for if(size > PARALLEL_THRESHOLD)
    for(i=0; i<size; i++) W->data[i] -= (lr * G->data[i]);
}

RING_FUNC(ring_tensor_dropout) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double rate = RING_API_GETNUMBER(2);
    double scale = 1.0 / (1.0 - rate);
    int i;
    // Serial
    for(i=0; i<T->size; i++) {
        if ((double)rand() / RAND_MAX < rate) T->data[i] = 0.0;
        else T->data[i] *= scale;
    }
}

/* ==================================================================== */
/* --- 7. UTILS ------------------------------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_get_cores) {
    int cores = 1;
    #ifdef _OPENMP
    cores = omp_get_num_procs();
    #endif
    RING_API_RETNUMBER(cores);
}

RING_FUNC(ring_tensor_set_threads) {
    int n = (int)RING_API_GETNUMBER(1);
    #ifdef _OPENMP
    omp_set_num_threads(n);
    #endif
}

/*
** ====================================================================
** --- CROSS ENTROPY KERNELS (FUSED) ----------------------------------
** ====================================================================
*/

/*
** Optimized CrossEntropy Loss (Fused Kernel)
** Logic: -Sum(Target * Log(Pred)) / ActiveSamples
** Features: Auto-Masking (Ignores zero targets), Safety Clamp (No NaN)
*/
RING_FUNC(ring_tensor_crossentropy_loss) {
    tensor_t *Probs   = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *Targets = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    
    double total_loss = 0.0;
    int active_samples = 0;
    double eps = 1e-7; // Safety Clamp

    int rows = Probs->rows;
    int cols = Probs->cols;
    int r, c;

    // Parallel Reduction for Speed
    #pragma omp parallel for reduction(+:total_loss, active_samples) private(c)
    for (r = 0; r < rows; r++) {
        int target_idx = -1;
        double *row_t = &Targets->data[r * cols];
        
        // Find Target Class (One-Hot)
        for (c = 0; c < cols; c++) {
            if (row_t[c] > 0.5) {
                target_idx = c;
                break;
            }
        }
        
        // If Target exists (Not Padding)
        if (target_idx != -1) {
            double p = Probs->data[r * cols + target_idx];
            
            // --- Safety Clamp (Same as your Ring code) ---
            if (p < eps) p = eps;
            if (p > 1.0) p = 1.0;
            // ---------------------------------------------
            
            total_loss += -log(p);
            active_samples++;
        }
    }

    if (active_samples == 0) {
        RING_API_RETNUMBER(0.0);
    } else {
        RING_API_RETNUMBER(total_loss / active_samples);
    }
}

/*
** Optimized Backward (Fused Kernel)
** Logic: (Probs - Targets) / ActiveSamples
** Features: Auto-Masking (Zeros out gradients for padding)
*/
RING_FUNC(ring_tensor_crossentropy_backward) {
    tensor_t *Probs   = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *Targets = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *Grad    = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);

    int rows = Probs->rows;
    int cols = Probs->cols;
    int active_samples = 0;
    int r, c;

    // 1. Count Active Samples First (For Scaling)
    for (r = 0; r < rows; r++) {
        double *row_t = &Targets->data[r * cols];
        for (c = 0; c < cols; c++) {
            if (row_t[c] > 0.5) {
                active_samples++;
                break;
            }
        }
    }
    
    double scale = (active_samples > 0) ? (1.0 / active_samples) : 0.0;

    // 2. Compute Gradients
    #pragma omp parallel for private(c)
    for (r = 0; r < rows; r++) {
        double *p_row = &Probs->data[r * cols];
        double *t_row = &Targets->data[r * cols];
        double *g_row = &Grad->data[r * cols];
        
        // Check mask
        int is_active = 0;
        for(c=0; c<cols; c++) if(t_row[c] > 0.5) { is_active=1; break; }

        if (is_active) {
            // Apply Gradient: (P - T) * Scale
            for (c = 0; c < cols; c++) {
                g_row[c] = (p_row[c] - t_row[c]) * scale;
            }
        } else {
            // Masking: Force Zero
            memset(g_row, 0, cols * sizeof(double));
        }
    }
}


/*
** Bulk Set from List (Turbo Loading)
** Copies a Ring List directly into Tensor Memory.
** Eliminates the overhead of calling setVal() thousands of times.
*/
RING_FUNC(ring_tensor_set_from_list) {
    tensor_t *T;
    List *pList;
    int i, nSize, nLimit;

    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    // 1. Get Tensor
    T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    
    // 2. Get List
    if (!RING_API_ISLIST(2)) {
        RING_API_ERROR(RING_API_BADPARATYPE);
        return;
    }
    pList = RING_API_GETLIST(2);
    
    // 3. Determine safe copy limit
    nSize = ring_list_getsize(pList);
    nLimit = T->size;
    if (nSize < nLimit) nLimit = nSize;

    // 4. Turbo Copy Loop (Inside C)
    for(i = 1; i <= nLimit; i++) {
        if (ring_list_isnumber(pList, i)) {
            T->data[i-1] = ring_list_getdouble(pList, i);
        } else {
            T->data[i-1] = 0.0; // Default for non-numbers
        }
    }
}

/*
** Set One-Hot (Scatter)
** Takes a List of Indices and sets T[row, index] = value.
** Assumes Tensor is already zeroed (or overwrites).
** 1-based Indexing for both List and Values.
*/
RING_FUNC(ring_tensor_set_one_hot) {
    tensor_t *T;
    List *pList;
    double val;
    int i, nListSize, nMaxRows;

    if (RING_API_PARACOUNT != 3) {
        RING_API_ERROR(RING_API_MISS3PARA);
        return;
    }

    T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    
    if (!RING_API_ISLIST(2)) {
        RING_API_ERROR("Param 2 must be a List of Indices");
        return;
    }
    pList = RING_API_GETLIST(2);
    val = RING_API_GETNUMBER(3);

    nListSize = ring_list_getsize(pList);
    nMaxRows = T->rows;
    
    if (nListSize > nMaxRows) nListSize = nMaxRows;

    // Parallel scatter
    #pragma omp parallel for if(nListSize > 5000)
    for(i = 1; i <= nListSize; i++) {
        if (ring_list_isnumber(pList, i)) {
            int col_idx = (int)ring_list_getdouble(pList, i);
            
            // ignore 0 (padding marker) و 1 (PAD token)
            if (col_idx <= 1) continue;
            
            // Validate bounds
            if (col_idx >= 2 && col_idx <= T->cols) {
                T->data[(i-1) * T->cols + (col_idx-1)] = val;
            }
        }
    }
}

/* ==================================================================== */
/* --- 8. PERSISTENCE (BINARY & QUANTIZATION) ------------------------- */
/* ==================================================================== */

/*
** Save Tensor (Raw Binary - Double Precision 64-bit)
** Format: [Rows (int)][Cols (int)][Data (double...)]
*/
RING_FUNC(ring_tensor_save) {
    tensor_t *t;
    const char *cFile;
    FILE *fp;
    
    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    cFile = RING_API_GETSTRING(2);

    fp = fopen(cFile, "wb");
    if (!fp) {
        RING_API_ERROR("Could not open file for writing");
        return;
    }

    // 1. Write Header (Rows, Cols)
    fwrite(&t->rows, sizeof(int), 1, fp);
    fwrite(&t->cols, sizeof(int), 1, fp);

    // 2. Write Data Block (Fastest way)
    fwrite(t->data, sizeof(double), t->size, fp);

    fclose(fp);
}

/*
** Load Tensor (Raw Binary - Double Precision)
** Returns: New Tensor Pointer
*/
RING_FUNC(ring_tensor_load) {
    const char *cFile;
    FILE *fp;
    int rows, cols;
    tensor_t *t;

    if (RING_API_PARACOUNT != 1) {
        RING_API_ERROR(RING_API_MISS1PARA);
        return;
    }

    cFile = RING_API_GETSTRING(1);
    fp = fopen(cFile, "rb");
    if (!fp) {
        RING_API_ERROR("Could not open file for reading");
        return;
    }

    // 1. Read Header
    if (fread(&rows, sizeof(int), 1, fp) != 1 ||
        fread(&cols, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        RING_API_ERROR("Invalid Tensor File (Header)");
        return;
    }

    // 2. Allocate Tensor
    t = (tensor_t *)malloc(sizeof(tensor_t));
    t->rows = rows;
    t->cols = cols;
    t->size = rows * cols;
    t->ndim = 2; // Default
    t->shape[0]=1; t->shape[1]=1; t->shape[2]=rows; t->shape[3]=cols;

    t->data = (double *)malloc(t->size * sizeof(double));
    if (!t->data) {
        fclose(fp);
        free(t);
        RING_API_ERROR("Malloc Failed");
        return;
    }

    // 3. Read Data Block
    if (fread(t->data, sizeof(double), t->size, fp) != t->size) {
        fclose(fp);
        free(t->data);
        free(t);
        RING_API_ERROR("File truncated or corrupted");
        return;
    }

    fclose(fp);
    RING_API_RETMANAGEDCPOINTER(t, RING_VM_POINTER_TENSOR, ring_tensor_free);
}

/*
** Save Tensor Quantized (FP32 - 32-bit Float)
** Reduces size by 50%.
*/
RING_FUNC(ring_tensor_save_fp32) {
    tensor_t *t;
    const char *cFile;
    FILE *fp;
    int i;
    float val; // 32-bit buffer

    if (RING_API_PARACOUNT != 2) return;

    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    cFile = RING_API_GETSTRING(2);

    fp = fopen(cFile, "wb");
    if (!fp) return;

    // Header
    fwrite(&t->rows, sizeof(int), 1, fp);
    fwrite(&t->cols, sizeof(int), 1, fp);

    // Write Data (Cast Double -> Float)
    // Optimization: We could use a buffer block, but loop is simple for now
    for (i = 0; i < t->size; i++) {
        val = (float)t->data[i];
        fwrite(&val, sizeof(float), 1, fp);
    }

    fclose(fp);
}

/*
** Load Tensor Quantized (FP32 -> Double)
*/
RING_FUNC(ring_tensor_load_fp32) {
    const char *cFile;
    FILE *fp;
    int rows, cols, i;
    tensor_t *t;
    float val;

    if (RING_API_PARACOUNT != 1) return;

    cFile = RING_API_GETSTRING(1);
    fp = fopen(cFile, "rb");
    if (!fp) return;

    fread(&rows, sizeof(int), 1, fp);
    fread(&cols, sizeof(int), 1, fp);

    t = (tensor_t *)malloc(sizeof(tensor_t));
    t->rows = rows; t->cols = cols; t->size = rows * cols;
    t->ndim = 2; t->shape[0]=1; t->shape[1]=1; t->shape[2]=rows; t->shape[3]=cols;
    t->data = (double *)malloc(t->size * sizeof(double));

    // Read Data (Float -> Double)
    for (i = 0; i < t->size; i++) {
        fread(&val, sizeof(float), 1, fp);
        t->data[i] = (double)val;
    }

    fclose(fp);
    RING_API_RETMANAGEDCPOINTER(t, RING_VM_POINTER_TENSOR, ring_tensor_free);
}

/*
** Load Tensor In-Place (Modifies existing tensor)
** Usage: tensor.loadFile(filename)
*/
RING_FUNC(ring_tensor_load_inplace) {
    tensor_t *t;
    const char *cFile;
    FILE *fp;
    int rows, cols, size;

    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    cFile = RING_API_GETSTRING(2);

    fp = fopen(cFile, "rb");
    if (!fp) {
        RING_API_ERROR("Could not open file for reading");
        return;
    }

    // Read header
    if (fread(&rows, sizeof(int), 1, fp) != 1 ||
        fread(&cols, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        RING_API_ERROR("Invalid file header");
        return;
    }

    size = rows * cols;

    // Free old data if size changed
    if (t->size != size) {
        if (t->data) {
            free(t->data);
        }
        t->data = (double *)malloc(size * sizeof(double));
        if (!t->data) {
            fclose(fp);
            RING_API_ERROR("Memory allocation failed");
            return;
        }
    }

    // Update dimensions
    t->rows = rows;
    t->cols = cols;
    t->size = size;
    t->ndim = 2;
    t->shape[0] = 1;
    t->shape[1] = 1;
    t->shape[2] = rows;
    t->shape[3] = cols;

    // Read data
    if (fread(t->data, sizeof(double), t->size, fp) != t->size) {
        fclose(fp);
        RING_API_ERROR("File read error");
        return;
    }

    fclose(fp);
    
    // No return - modified in place
}

/*
** Load Quantized Tensor In-Place (FP32 -> Double)
*/
RING_FUNC(ring_tensor_load_fp32_inplace) {
    tensor_t *t;
    const char *cFile;
    FILE *fp;
    int rows, cols, size, i;
    float val;

    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    cFile = RING_API_GETSTRING(2);

    fp = fopen(cFile, "rb");
    if (!fp) {
        RING_API_ERROR("Could not open file");
        return;
    }

    // Read header
    fread(&rows, sizeof(int), 1, fp);
    fread(&cols, sizeof(int), 1, fp);
    size = rows * cols;

    // Reallocate if needed
    if (t->size != size) {
        if (t->data) free(t->data);
        t->data = (double *)malloc(size * sizeof(double));
    }

    // Update dimensions
    t->rows = rows;
    t->cols = cols;
    t->size = size;
    t->ndim = 2;
    t->shape[0] = 1;
    t->shape[1] = 1;
    t->shape[2] = rows;
    t->shape[3] = cols;

    // Read data (float -> double)
    for (i = 0; i < t->size; i++) {
        fread(&val, sizeof(float), 1, fp);
        t->data[i] = (double)val;
    }

    fclose(fp);
}
/*
** Get Tensor Dimensions (For Syncing with Ring)
*/
RING_FUNC(ring_tensor_get_rows) {
    tensor_t *t;
    if (RING_API_PARACOUNT != 1) {
        RING_API_ERROR(RING_API_MISS1PARA);
        return;
    }
    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    RING_API_RETNUMBER(t->rows);
}

/*
** Get Tensor Columns
*/
RING_FUNC(ring_tensor_get_cols) {
    tensor_t *t;
    if (RING_API_PARACOUNT != 1) {
        RING_API_ERROR(RING_API_MISS1PARA);
        return;
    }
    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    RING_API_RETNUMBER(t->cols);
}

/*
** Global Gradient Clipping (Fixed Logic)
** Applies the same "Pointer OR Number" check to sub-lists.
*/
RING_FUNC(ring_tensor_clip_global_norm) {
    List *pList, *pSubList;
    double max_norm, total_norm_sq = 0.0;
    int i, k, nItems;
    tensor_t *t;

    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }

    if (!RING_API_ISLIST(1)) {
        RING_API_ERROR("Param 1 must be a List");
        return;
    }

    pList = RING_API_GETLIST(1);
    max_norm = RING_API_GETNUMBER(2);
    nItems = ring_list_getsize(pList);

    // ═══════════════════════════════════════════════
    // STEP 1: Calculate Global Norm
    // ═══════════════════════════════════════════════
    int valid_tensors = 0;  
    
    for (i = 1; i <= nItems; i++) {
        t = NULL;

        if (ring_list_islist(pList, i)) {
            pSubList = ring_list_getlist(pList, i);
            if (ring_list_getsize(pSubList) > 0) {
                if (ring_list_ispointer(pSubList, 1)) {
                    t = (tensor_t *)ring_list_getpointer(pSubList, 1);
                } 
                else if (ring_list_isnumber(pSubList, 1)) {
                    t = (tensor_t *)(size_t)ring_list_getdouble(pSubList, 1);
                }
            }
        } 
        else if (ring_list_ispointer(pList, i)) {
            t = (tensor_t *)ring_list_getpointer(pList, i);
        }
        else if (ring_list_isnumber(pList, i)) {
            t = (tensor_t *)(size_t)ring_list_getdouble(pList, i);
        }

        if (t == NULL) continue;
        
        valid_tensors++;  

        double t_sum = 0.0;
        int size = t->size;
        double *data = t->data;
        
        #pragma omp parallel for reduction(+:t_sum) if(size > 5000)
        for (k = 0; k < size; k++) {
            t_sum += (data[k] * data[k]);
        }
        total_norm_sq += t_sum;
    }

    // ═══════════════════════════════════════════════
    // STEP 2: Calculate Scale and Clip
    // ═══════════════════════════════════════════════
    double total_norm = sqrt(total_norm_sq);
    double scale = 1.0;
    int clipped = 0;  
    
    if (total_norm > max_norm) {
        scale = max_norm / (total_norm + 1e-9);
        clipped = 1;
        
        // Apply scale to all tensors
        for (i = 1; i <= nItems; i++) {
            t = NULL;
            
            if (ring_list_islist(pList, i)) {
                pSubList = ring_list_getlist(pList, i);
                if (ring_list_getsize(pSubList) > 0) {
                    if (ring_list_ispointer(pSubList, 1)) {
                        t = (tensor_t *)ring_list_getpointer(pSubList, 1);
                    } 
                    else if (ring_list_isnumber(pSubList, 1)) {
                        t = (tensor_t *)(size_t)ring_list_getdouble(pSubList, 1);
                    }
                }
            }
            else if (ring_list_ispointer(pList, i)) {
                t = (tensor_t *)ring_list_getpointer(pList, i);
            }
            else if (ring_list_isnumber(pList, i)) {
                t = (tensor_t *)(size_t)ring_list_getdouble(pList, i);
            }

            if (t != NULL) {
                int size = t->size;
                double *data = t->data;
                
                #pragma omp parallel for if(size > 5000)
                for (k = 0; k < size; k++) {
                    data[k] *= scale;
                }
            }
        }
    }
    
    //  Debug output
    //if (clipped) {
    //    printf("[CLIP] Norm %.2f > %.2f → scaled by %.4f (tensors: %d)\n", 
    //           total_norm, max_norm, scale, valid_tensors);
    //}
    
    RING_API_RETNUMBER(total_norm);
}

/*
** GELU Activation (Gaussian Error Linear Unit)
** Used in GPT/BERT. Smooth non-linearity.
** Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
*/
RING_FUNC(ring_tensor_gelu) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    int size = T->size;
    double x, cube, inner;
    
    // Constants
    const double SQRT_2_OVER_PI = 0.7978845608;
    const double COEF = 0.044715;

    #pragma omp parallel for if(size > 2000) private(x, cube, inner)
    for(i=0; i<size; i++) {
        x = T->data[i];
        cube = x * x * x;
        inner = SQRT_2_OVER_PI * (x + COEF * cube);
        T->data[i] = 0.5 * x * (1.0 + tanh(inner));
    }
}

/*
** GELU Derivative (Gradient)
** Backward pass for GELU.
** Formula is complex, essentially: 0.5 [1 + tanh(y) + x * (1 - tanh(y)^2) * dy/dx]
*/
RING_FUNC(ring_tensor_gelu_prime) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    int size = T->size;
    double x, x3, inner, tanh_inner, sech2;
    
    const double SQRT_2_OVER_PI = 0.7978845608;
    const double COEF = 0.044715;

    #pragma omp parallel for if(size > 2000) private(x, x3, inner, tanh_inner, sech2)
    for(i=0; i<size; i++) {
        x = T->data[i];
        x3 = x * x * x;
        inner = SQRT_2_OVER_PI * (x + COEF * x3);
        tanh_inner = tanh(inner);
        
        // Sech^2(x) = 1 - tanh^2(x)
        sech2 = 1.0 - (tanh_inner * tanh_inner);
        
        // Derivative logic
        double term1 = 0.5 * (1.0 + tanh_inner);
        double term2 = 0.5 * x * sech2 * SQRT_2_OVER_PI * (1.0 + 3.0 * COEF * x * x);
        
        T->data[i] = term1 + term2;
    }
}

/*
** Sum of Squares (Helper for L2 Norm Calculation)
** Returns: Sum(x^2)
*/
RING_FUNC(ring_tensor_sum_squares) {
    tensor_t *T;
    double sum = 0.0;
    int i;

    if (RING_API_PARACOUNT != 1) {
        RING_API_ERROR(RING_API_MISS1PARA);
        return;
    }

    T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    
    // OpenMP Reduction for Speed
    #pragma omp parallel for reduction(+:sum) if(T->size > 2000)
    for(i = 0; i < T->size; i++) {
        sum += (T->data[i] * T->data[i]);
    }
    
    RING_API_RETNUMBER(sum);
}

/*
** Clip Tensor Values (In-place)
** Clips all values in tensor to [-max_val, +max_val]
*/
RING_FUNC(ring_tensor_clip_tensor) {
    tensor_t *T;
    double max_val;
    int i;
    
    if (RING_API_PARACOUNT != 2) {
        RING_API_ERROR(RING_API_MISS2PARA);
        return;
    }
    
    T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    max_val = RING_API_GETNUMBER(2);
    
    #pragma omp parallel for if(T->size > 1000)
    for(i = 0; i < T->size; i++) {
        if (T->data[i] > max_val) {
            T->data[i] = max_val;
        } else if (T->data[i] < -max_val) {
            T->data[i] = -max_val;
        }
    }
}

/*
** Simplified Multi-Head Attention Backward
** Fast approximation that just redistributes gradients with proper scaling
*/
RING_FUNC(ring_tensor_mha_backward_fast) {
    tensor_t *grad_concat, *grad_input;
    int num_heads;
    double clip_norm;
    int i;
    
    if (RING_API_PARACOUNT != 4) {
        RING_API_ERROR("Usage: mha_backward_fast(grad_concat, grad_input, num_heads, clip_norm)");
        return;
    }
    
    grad_concat = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    grad_input  = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    num_heads   = (int)RING_API_GETNUMBER(3);
    clip_norm   = RING_API_GETNUMBER(4);
    
    // ════════════════════════════════════════════════
    // STEP 1: Clip concat gradient
    // ════════════════════════════════════════════════
    double norm_sq = 0.0;
    
    #pragma omp parallel for reduction(+:norm_sq)
    for(i = 0; i < grad_concat->size; i++) {
        norm_sq += grad_concat->data[i] * grad_concat->data[i];
    }
    
    double norm = sqrt(norm_sq);
    
    if (norm > clip_norm) {
        double scale = clip_norm / (norm + 1e-8);
        #pragma omp parallel for
        for(i = 0; i < grad_concat->size; i++) {
            grad_concat->data[i] *= scale;
        }
    }
    
    // ════════════════════════════════════════════════
    // STEP 2: Simple gradient redistribution
    // Scale by 1/sqrt(num_heads * 3) and copy
    // ════════════════════════════════════════════════
    double dist_scale = 1.0 / sqrt((double)(num_heads * 3));
    
    // Assuming grad_input and grad_concat have same size (simplified)
    // In practice, there's a projection but we approximate here
    
    int min_size = (grad_concat->size < grad_input->size) ? 
                   grad_concat->size : grad_input->size;
    
    #pragma omp parallel for
    for(i = 0; i < min_size; i++) {
        // Distribute to input (approximating Q, K, V backward)
        grad_input->data[i] = grad_concat->data[i] * dist_scale * 3.0;
    }
    
    // ════════════════════════════════════════════════
    // STEP 3: Clip output gradient
    // ════════════════════════════════════════════════
    norm_sq = 0.0;
    
    #pragma omp parallel for reduction(+:norm_sq)
    for(i = 0; i < grad_input->size; i++) {
        norm_sq += grad_input->data[i] * grad_input->data[i];
    }
    
    norm = sqrt(norm_sq);
    
    if (norm > clip_norm) {
        double scale = clip_norm / (norm + 1e-8);
        #pragma omp parallel for
        for(i = 0; i < grad_input->size; i++) {
            grad_input->data[i] *= scale;
        }
    }
    
    // Return final norm for monitoring
    RING_API_RETNUMBER(norm);
}

/*
** Repeat Tensor Rows (Tiling/Broadcasting)
** Copies the entire Src tensor content 'nTimes' into Dest.
** Used for Positional Embeddings Broadcasting.
** Optimization: Uses Block Memcpy.
*/
RING_FUNC(ring_tensor_repeat_rows) {
    tensor_t *Src, *Dest;
    int nTimes;
    int i;
    size_t chunk_bytes;

    if (RING_API_PARACOUNT != 3) {
        RING_API_ERROR(RING_API_MISS3PARA);
        return;
    }

    Src = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    Dest = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    nTimes = (int)RING_API_GETNUMBER(3);

    // Validation
    if (Dest->size != Src->size * nTimes) {
        RING_API_ERROR("Repeat Rows: Destination size mismatch");
        return;
    }

    // Calculation
    chunk_bytes = Src->size * sizeof(double);
    
    // Execution: Parallel Block Copy
    // If nTimes is large (large batch), we parallelize the copy
    #pragma omp parallel for if(nTimes > 4)
    for (i = 0; i < nTimes; i++) {
        // Calculate offset for this block
        double *dest_ptr = &Dest->data[i * Src->size];
        
        // Copy the source block
        memcpy(dest_ptr, Src->data, chunk_bytes);
    }
}

/*
** Create Tensor from Raw Memory Address
** Usage: tensor_from_memory(nAddress, nRows, nCols)
*/
RING_FUNC(ring_tensor_from_memory) {
    if (RING_API_PARACOUNT != 3) {
        RING_API_ERROR(RING_API_MISS3PARA);
        return;
    }

    // 1. Get Address as Number (size_t)
    size_t address = (size_t)RING_API_GETNUMBER(1);
    int rows = (int)RING_API_GETNUMBER(2);
    int cols = (int)RING_API_GETNUMBER(3);

    // 2. Allocate Struct Wrapper
    tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
    if (!t) { RING_API_ERROR("Malloc Fail"); return; }

    // 3. Point to External Memory
    t->data = (double *)address;
    t->rows = rows;
    t->cols = cols;
    t->size = rows * cols;
    t->ndim = 2;
    t->shape[0]=1; t->shape[1]=1; t->shape[2]=rows; t->shape[3]=cols;

    // 4. Return with Special Destructor
    RING_API_RETMANAGEDCPOINTER(t, RING_VM_POINTER_TENSOR, ring_tensor_free_struct_only);
}

/*
** Linear Causal Attention (Recurrent / The Geomancy Kernel)
** Logic: S_i = S_{i-1} + (K_i * V_i^T)
**        O_i = (Q_i * S_i) / (Q_i * Z_i)
**
** This guarantees that token 'i' cannot see 'i+1'.
*/
RING_FUNC(ring_tensor_attention_linear_causal) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    
    // Params: Scale is implied or pre-applied to Q/K? 
    // Usually Linear Attention doesn't use scale 1/sqrt(d), but feature map phi(x).
    // We assume Q and K are already passed through ReLU.
    // If you pass scale, we can apply it.
    double scale = 1.0; 
    if (RING_API_PARACOUNT >= 5) scale = RING_API_GETNUMBER(5);

    int batch = 1; 
    if (RING_API_PARACOUNT >= 6) batch = (int)RING_API_GETNUMBER(6);

    int seq_len = Q->rows / batch;
    int dim = Q->cols;
    
    int b, t, i, j;
    double eps = 1e-6;

    // Parallelize over Batches (Safe)
    #pragma omp parallel for private(b, t, i, j)
    for (b = 0; b < batch; b++) {
        
        // Offset for current batch
        int offset = b * seq_len * dim;
        
        // State Matrix S (dim x dim) - Initialized to 0
        double *S = (double *)calloc(dim * dim, sizeof(double));
        // Normalizer Z (dim) - Initialized to 0
        double *Z = (double *)calloc(dim, sizeof(double));
        
        if (!S || !Z) continue; // Safety skip

        // Iterate Time Steps (Sequential per batch)
        for (t = 0; t < seq_len; t++) {
            
            double *qt = &Q->data[offset + t*dim];
            double *kt = &K->data[offset + t*dim];
            double *vt = &V->data[offset + t*dim];
            double *ot = &Out->data[offset + t*dim];
            
            // 1. Update State: S += K_t^T * V_t
            // Also Update Z += K_t
            for (i = 0; i < dim; i++) {
                double k_val = kt[i];
                Z[i] += k_val;
                
                for (j = 0; j < dim; j++) {
                    S[i * dim + j] += k_val * vt[j];
                }
            }
            
            // 2. Compute Output: O_t = (Q_t * S) / (Q_t * Z)
            double den = 0.0;
            for (i = 0; i < dim; i++) den += qt[i] * Z[i];
            
            if (den < eps) den = eps; // Avoid div/0
            
            for (j = 0; j < dim; j++) { // For each output dimension
                double num = 0.0;
                for (i = 0; i < dim; i++) {
                    // num += qt[i] * S[i][j]
                    num += qt[i] * S[i * dim + j];
                }
                ot[j] = (num / den) * scale; // Apply scale if needed
            }
        }
        
        free(S);
        free(Z);
    }
}

/*
** Optimized Linear Attention (Matrix Form + Tiling)
** Formula: O = Q * (K^T * V)
** Complexity: O(N) instead of O(N^2)
** Optimization: Tiled Loops for Cache Locality
*/ 
RING_FUNC(ring_tensor_attention_linear_optimized) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    double scale = RING_API_GETNUMBER(5);
    int batch = (int)RING_API_GETNUMBER(6); // Explicit Batch
    
    int seq = Q->rows / batch;
    int dim = Q->cols;
    
    // Pointers
    int b, i, j, k, ii, jj, kk;
    
    // Parallelize over Batches
    #pragma omp parallel for private(b, i, j, k, ii, jj, kk)
    for (b = 0; b < batch; b++) {
        
        // 1. Allocate Global Context Matrix (d x d)
        // This replaces the N x N matrix. Ideally small (64x64).
        double *Context = (double *)calloc(dim * dim, sizeof(double));
        
        // Offsets for this batch
        double *qb = &Q->data[b * seq * dim];
        double *kb = &K->data[b * seq * dim];
        double *vb = &V->data[b * seq * dim];
        double *ob = &Out->data[b * seq * dim];

        // --- STEP 1: Compute Context = K^T * V ---
        // Shape: (d, N) * (N, d) -> (d, d)
        // TILED IMPLEMENTATION
        
        for (ii = 0; ii < dim; ii += TILE_SIZE) {
            for (kk = 0; kk < seq; kk += TILE_SIZE) {
                for (jj = 0; jj < dim; jj += TILE_SIZE) {
                    
                    // Mini-Block Processing
                    int i_lim = (ii + TILE_SIZE < dim) ? ii + TILE_SIZE : dim;
                    int k_lim = (kk + TILE_SIZE < seq) ? kk + TILE_SIZE : seq;
                    int j_lim = (jj + TILE_SIZE < dim) ? jj + TILE_SIZE : dim;

                    for (i = ii; i < i_lim; i++) {
                        for (k = kk; k < k_lim; k++) {
                            double k_val = kb[k * dim + i]; // Transposed access
                            
                            for (j = jj; j < j_lim; j++) {
                                Context[i * dim + j] += k_val * vb[k * dim + j];
                            }
                        }
                    }
                }
            }
        }

        // --- STEP 2: Compute Output = Q * Context ---
        // Shape: (N, d) * (d, d) -> (N, d)
        // TILED IMPLEMENTATION
        
        for (ii = 0; ii < seq; ii += TILE_SIZE) {
            for (kk = 0; kk < dim; kk += TILE_SIZE) {
                for (jj = 0; jj < dim; jj += TILE_SIZE) {
                    
                    int i_lim = (ii + TILE_SIZE < seq) ? ii + TILE_SIZE : seq;
                    int k_lim = (kk + TILE_SIZE < dim) ? kk + TILE_SIZE : dim;
                    int j_lim = (jj + TILE_SIZE < dim) ? jj + TILE_SIZE : dim;
                    
                    for (i = ii; i < i_lim; i++) {
                        for (k = kk; k < k_lim; k++) {
                            double q_val = qb[i * dim + k];
                            
                            for (j = jj; j < j_lim; j++) {
                                ob[i * dim + j] += (q_val * Context[k * dim + j]);
                            }
                        }
                    }
                }
            }
        }
        
        // Apply Scale
        if (scale != 1.0) {
            for(i=0; i < seq*dim; i++) ob[i] *= scale;
        }

        free(Context);
    }
}

/*
** Optimized Backward for Linear Causal Attention
** Formula: O = Q * (CumSum(K^T * V))
** Recomputes 'S' on the fly to save memory.
*/
RING_FUNC(ring_tensor_attention_linear_backward) {
    tensor_t *Q  = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K  = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V  = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *G  = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR); // Grad Output
    
    tensor_t *dQ = (tensor_t *)RING_API_GETCPOINTER(5, RING_VM_POINTER_TENSOR);
    tensor_t *dK = (tensor_t *)RING_API_GETCPOINTER(6, RING_VM_POINTER_TENSOR);
    tensor_t *dV = (tensor_t *)RING_API_GETCPOINTER(7, RING_VM_POINTER_TENSOR);
    
    double scale = RING_API_GETNUMBER(8);
    int batch    = (int)RING_API_GETNUMBER(9);
    
    int dim = Q->cols;
    int seq = Q->rows / batch;
    
    int b, t, i, j;
    
    // Parallelize over Batches
    #pragma omp parallel for private(b, t, i, j)
    for (b = 0; b < batch; b++) {
        
        int offset = b * seq * dim;
        
        // 1. Recompute States (Forward Pass) to store History
        // We need S_t at every step to calculate dQ_t
        // Size: Seq * Dim * Dim. This is RAM heavy but necessary for speed.
        double *S_History = (double *)malloc(seq * dim * dim * sizeof(double));
        
        // Temp State S
        double *S = (double *)calloc(dim * dim, sizeof(double));
        
        for (t = 0; t < seq; t++) {
            double *kt = &K->data[offset + t*dim];
            double *vt = &V->data[offset + t*dim];
            
            // S += k^T * v
            for (i = 0; i < dim; i++) {
                for (j = 0; j < dim; j++) {
                    S[i*dim + j] += kt[i] * vt[j];
                }
            }
            
            // Store S_t
            memcpy(&S_History[t*dim*dim], S, dim*dim*sizeof(double));
        }
        free(S);
        
        // 2. Backward Pass (Reverse Time)
        // dS accumulates gradients flowing back from future steps
        double *dS = (double *)calloc(dim * dim, sizeof(double));
        
        for (t = seq - 1; t >= 0; t--) {
            double *qt  = &Q->data[offset + t*dim];
            double *kt  = &K->data[offset + t*dim];
            double *vt  = &V->data[offset + t*dim];
            double *gt  = &G->data[offset + t*dim];
            
            double *dqt = &dQ->data[offset + t*dim];
            double *dkt = &dK->data[offset + t*dim];
            double *dvt = &dV->data[offset + t*dim];
            
            // Retrieve S at this step
            double *St = &S_History[t*dim*dim];
            
            // A. Compute dQ_t = G_t * S_t^T
            for (i = 0; i < dim; i++) {
                double val = 0.0;
                for (j = 0; j < dim; j++) {
                    val += gt[j] * St[i*dim + j]; // S is (dim, dim)
                }
                dqt[i] = val * scale;
            }
            
            // B. Update dS += G_t^T * Q_t
            for (i = 0; i < dim; i++) {
                for (j = 0; j < dim; j++) {
                    dS[i*dim + j] += qt[i] * gt[j];
                }
            }
            
            // C. Compute dK_t = dS * V_t
            for (i = 0; i < dim; i++) {
                double val = 0.0;
                for (j = 0; j < dim; j++) {
                    val += dS[i*dim + j] * vt[j];
                }
                dkt[i] = val * scale;
            }
            
            // D. Compute dV_t = dS^T * K_t
            for (i = 0; i < dim; i++) {
                double val = 0.0;
                for (j = 0; j < dim; j++) {
                    val += dS[j*dim + i] * kt[j];
                }
                dvt[i] = val * scale;
            }
        }
        
        free(S_History);
        free(dS);
    }
}

/*
** Multi-Head Attention Kernel (The All-In-One)
** Input: Q, K, V (Batch, Seq, Dim) -> where Dim = Heads * HeadDim
** Performs: Split Heads -> Attention -> Merge Heads
*/
RING_FUNC(ring_tensor_attention_multihead) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *Out = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    
    double scale = RING_API_GETNUMBER(5);
    int batch    = (int)RING_API_GETNUMBER(6);
    int seq      = (int)RING_API_GETNUMBER(7);
    int heads    = (int)RING_API_GETNUMBER(8);
    int is_causal= (int)RING_API_GETNUMBER(9);
    
    int dim = Q->cols; 
    int head_dim = dim / heads;
    
    if (dim % heads != 0) { RING_API_ERROR("Dim not divisible by heads"); return; }
    
    // تعريف المتغيرات خارج الحلقة ليتوافق مع MSVC
    int task, b, h, i, j, k;
    int total_tasks = batch * heads;
    
    #pragma omp parallel for private(task, b, h, i, j, k)
    for (task = 0; task < total_tasks; task++) {
        b = task / heads;
        h = task % heads;
        
        double *scores = (double *)malloc(seq * sizeof(double));
        
        if (scores) {
            int batch_offset = b * seq * dim;
            int head_offset  = h * head_dim; 
            
            for (i = 0; i < seq; i++) {
                double *qi = &Q->data[batch_offset + (i * dim) + head_offset];
                double *oi = &Out->data[batch_offset + (i * dim) + head_offset];
                
                // 1. Score
                for (j = 0; j < seq; j++) {
                    if (is_causal && j > i) {
                        scores[j] = -1e9;
                        continue;
                    }
                    
                    double *kj = &K->data[batch_offset + (j * dim) + head_offset];
                    double dot = 0.0;
                    for (k = 0; k < head_dim; k++) dot += qi[k] * kj[k];
                    scores[j] = dot * scale;
                }
                
                // 2. Softmax
                double maxVal = -1e9;
                for(j=0; j<seq; j++) if(scores[j] > maxVal) maxVal = scores[j];
                
                double sum = 0.0;
                for(j=0; j<seq; j++) {
                    scores[j] = exp(scores[j] - maxVal);
                    sum += scores[j];
                }
                double invSum = 1.0 / (sum + 1e-9);
                for(j=0; j<seq; j++) scores[j] *= invSum;
                
                // 3. Output
                for (k = 0; k < head_dim; k++) oi[k] = 0.0;
                
                for (j = 0; j < seq; j++) {
                    double s = scores[j];
                    if (s < 1e-9) continue;
                    
                    double *vj = &V->data[batch_offset + (j * dim) + head_offset];
                    for (k = 0; k < head_dim; k++) {
                        oi[k] += s * vj[k];
                    }
                }
            }
            free(scores);
        }
    }
}



/*
** Multi-Head Attention Backward (The Master Kernel)
** Recomputes Scores (S) to save memory, calculates exact gradients.
** Inputs: Q, K, V, GradOut
** Outputs: dQ, dK, dV
*/
RING_FUNC(ring_tensor_attention_multihead_backward) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *dOut = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    
    tensor_t *dQ = (tensor_t *)RING_API_GETCPOINTER(5, RING_VM_POINTER_TENSOR);
    tensor_t *dK = (tensor_t *)RING_API_GETCPOINTER(6, RING_VM_POINTER_TENSOR);
    tensor_t *dV = (tensor_t *)RING_API_GETCPOINTER(7, RING_VM_POINTER_TENSOR);
    
    double scale = RING_API_GETNUMBER(8);
    int batch    = (int)RING_API_GETNUMBER(9);
    int seq      = (int)RING_API_GETNUMBER(10);
    int heads    = (int)RING_API_GETNUMBER(11);
    int is_causal= (int)RING_API_GETNUMBER(12);
    
    int dim = Q->cols;
    int head_dim = dim / heads;
    
    // تعريف المتغيرات خارج الحلقة لتوافق MSVC
    int task, b, h, i, j, k;
    int total_tasks = batch * heads;

    #pragma omp parallel for private(task, b, h, i, j, k)
    for (task = 0; task < total_tasks; task++) {
        b = task / heads;
        h = task % heads;
        
        int offset = b * seq * dim;
        int head_off = h * head_dim;
        
        double *S  = (double *)malloc(seq * seq * sizeof(double));
        double *dS = (double *)malloc(seq * seq * sizeof(double));
        
        if (S && dS) {
            
            // --- 1. Recompute Forward ---
            for (i = 0; i < seq; i++) {
                double *qi = &Q->data[offset + i*dim + head_off];
                
                for (j = 0; j < seq; j++) {
                    double *kj = &K->data[offset + j*dim + head_off];
                    double dot = 0.0;
                    
                    if (is_causal && j > i) {
                        S[i*seq + j] = 0.0;
                        continue; 
                    }

                    for (k = 0; k < head_dim; k++) dot += qi[k] * kj[k];
                    S[i*seq + j] = dot * scale;
                }
                
                double maxVal = -1e9;
                for (j = 0; j <= (is_causal ? i : seq-1); j++) {
                    if (S[i*seq + j] > maxVal) maxVal = S[i*seq + j];
                }
                
                double sum = 0.0;
                for (j = 0; j <= (is_causal ? i : seq-1); j++) {
                    S[i*seq + j] = exp(S[i*seq + j] - maxVal);
                    sum += S[i*seq + j];
                }
                double invSum = 1.0 / (sum + 1e-9);
                for (j = 0; j <= (is_causal ? i : seq-1); j++) {
                    S[i*seq + j] *= invSum;
                }
            }

            // --- 2. Calculate dV ---
            for (j = 0; j < seq; j++) {
                double *dvj = &dV->data[offset + j*dim + head_off];
                for (k = 0; k < head_dim; k++) dvj[k] = 0.0;

                for (i = 0; i < seq; i++) {
                    if (is_causal && j > i) continue;
                    
                    double s_val = S[i*seq + j];
                    double *doi = &dOut->data[offset + i*dim + head_off];
                    
                    for (k = 0; k < head_dim; k++) {
                        dvj[k] += s_val * doi[k];
                    }
                }
            }

            // --- 3. Calculate dS ---
            for (i = 0; i < seq; i++) {
                double *doi = &dOut->data[offset + i*dim + head_off];
                double dp_sum = 0.0;
                
                for (j = 0; j < seq; j++) {
                    if (is_causal && j > i) { dS[i*seq+j] = 0; continue; }
                    
                    double *vj = &V->data[offset + j*dim + head_off];
                    double dot = 0.0;
                    for (k = 0; k < head_dim; k++) dot += doi[k] * vj[k];
                    
                    dS[i*seq + j] = dot;
                    dp_sum += dot * S[i*seq + j];
                }
                
                for (j = 0; j < seq; j++) {
                    if (is_causal && j > i) continue;
                    double s_val = S[i*seq + j];
                    dS[i*seq + j] = s_val * (dS[i*seq + j] - dp_sum) * scale;
                }
            }

            // --- 4. Calculate dQ and dK ---
            for (i = 0; i < seq; i++) {
                double *dqi = &dQ->data[offset + i*dim + head_off];
                for (k = 0; k < head_dim; k++) dqi[k] = 0.0;

                for (j = 0; j < seq; j++) {
                    if (is_causal && j > i) continue;
                    
                    double ds_val = dS[i*seq + j];
                    double *kj = &K->data[offset + j*dim + head_off];
                    
                    for (k = 0; k < head_dim; k++) {
                        dqi[k] += ds_val * kj[k];
                    }
                }
            }

            for (j = 0; j < seq; j++) {
                double *dkj = &dK->data[offset + j*dim + head_off];
                for (k = 0; k < head_dim; k++) dkj[k] = 0.0;

                for (i = 0; i < seq; i++) {
                    if (is_causal && j > i) continue;
                    
                    double ds_val = dS[i*seq + j];
                    double *qi = &Q->data[offset + i*dim + head_off];
                    
                    for (k = 0; k < head_dim; k++) {
                        dkj[k] += ds_val * qi[k];
                    }
                }
            }
            
            free(S);
            free(dS);
        }
    }
}

/*
** Linear Global Attention Backward
** Exact gradient calculation for O(N) attention.
** Recomputes Context (K^T * V) to save memory.
*/
RING_FUNC(ring_tensor_attention_linear_global_backward) {
    tensor_t *Q = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *K = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *V = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    tensor_t *dOut = (tensor_t *)RING_API_GETCPOINTER(4, RING_VM_POINTER_TENSOR);
    
    tensor_t *dQ = (tensor_t *)RING_API_GETCPOINTER(5, RING_VM_POINTER_TENSOR);
    tensor_t *dK = (tensor_t *)RING_API_GETCPOINTER(6, RING_VM_POINTER_TENSOR);
    tensor_t *dV = (tensor_t *)RING_API_GETCPOINTER(7, RING_VM_POINTER_TENSOR);
    
    double scale = RING_API_GETNUMBER(8);
    int batch    = (int)RING_API_GETNUMBER(9);
    
    int seq = Q->rows / batch;
    int dim = Q->cols;
    
    int b, t, i, j;

    #pragma omp parallel for private(b, t, i, j)
    for (b = 0; b < batch; b++) {
        int offset = b * seq * dim;
        
        // Temp Buffers (d x d)
        // C_mat = K^T * V
        // dC_mat = Q^T * dOut
        double *C_mat  = (double *)calloc(dim * dim, sizeof(double));
        double *dC_mat = (double *)calloc(dim * dim, sizeof(double));
        
        if (!C_mat || !dC_mat) continue; // Safety

        // --- 1. Recompute C = K^T * V ---
        // And simultaneously partial compute dC = Q^T * dOut
        // We iterate sequence once to build these small matrices
        
        for (t = 0; t < seq; t++) {
            double *kt = &K->data[offset + t*dim];
            double *vt = &V->data[offset + t*dim];
            double *qt = &Q->data[offset + t*dim];
            double *gt = &dOut->data[offset + t*dim];
            
            for (i = 0; i < dim; i++) {
                for (j = 0; j < dim; j++) {
                    // C[i][j] += K[t][i] * V[t][j]
                    C_mat[i*dim + j] += kt[i] * vt[j];
                    
                    // dC[i][j] += Q[t][i] * G[t][j]
                    dC_mat[i*dim + j] += qt[i] * gt[j];
                }
            }
        }
        
        // --- 2. Compute Gradients ---
        for (t = 0; t < seq; t++) {
            double *dqt = &dQ->data[offset + t*dim];
            double *dkt = &dK->data[offset + t*dim];
            double *dvt = &dV->data[offset + t*dim];
            
            double *vt = &V->data[offset + t*dim];
            double *kt = &K->data[offset + t*dim];
            double *gt = &dOut->data[offset + t*dim];

            for (i = 0; i < dim; i++) {
                
                // dQ = dOut * C^T
                double sum_dq = 0.0;
                for (j = 0; j < dim; j++) {
                    sum_dq += gt[j] * C_mat[j*dim + i]; // C[j][i] is C^T
                }
                dqt[i] = sum_dq * scale;
                
                // dK = V * dC^T
                double sum_dk = 0.0;
                for (j = 0; j < dim; j++) {
                    sum_dk += vt[j] * dC_mat[j*dim + i]; // dC[j][i]
                }
                dkt[i] = sum_dk * scale;
                
                // dV = K * dC
                double sum_dv = 0.0;
                for (j = 0; j < dim; j++) {
                    sum_dv += kt[j] * dC_mat[i*dim + j];
                }
                dvt[i] = sum_dv * scale;
            }
        }
        
        free(C_mat);
        free(dC_mat);
    }
}

/*
** Set One-Hot from Raw Pointer (Turbo Scatter)
** Reads indices directly from AlQalam memory (C++ vector<double>) 
** and sets T[row, col] = val.
** Params: 1:Tensor, 2:RawPtrAddress, 3:Count, 4:Value
*/
RING_FUNC(ring_tensor_set_one_hot_ptr) {
    tensor_t *T;
    double *pIndices;
    int nCount, i;
    double val;

    if (RING_API_PARACOUNT != 4) {
        RING_API_ERROR(RING_API_MISS4PARA);
        return;
    }

    T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    
    // Get the raw memory address passed from Ring (AlQalam.getRawPointer)
    size_t ptrAddr = (size_t)RING_API_GETNUMBER(2);
    pIndices = (double *)ptrAddr;
    
    nCount = (int)RING_API_GETNUMBER(3);
    val = RING_API_GETNUMBER(4);

    if (pIndices == NULL) return;
    
    // Limit to Tensor rows to prevent overflow
    // We assume the tensor is flattened (Batch*Seq, Vocab)
    if (nCount > T->rows) nCount = T->rows;

    // Parallel Scatter
    #pragma omp parallel for if(nCount > 5000)
    for(i = 0; i < nCount; i++) {
        // Read index directly from C++ memory
        int col_idx = (int)pIndices[i];
        
        // Validate Bounds (1-based index)
        if (col_idx >= 1 && col_idx <= T->cols) {
            // Row i, Col (col_idx-1)
            T->data[i * T->cols + (col_idx - 1)] = val;
        }
    }
}

/*
** Get Raw Data Pointer
** Returns the actual memory address of the array as a double number to be passed to AlQalam
*/
RING_FUNC(ring_tensor_get_data_ptr) {
    tensor_t *t;
    if (RING_API_PARACOUNT != 1) {
        RING_API_ERROR(RING_API_MISS1PARA);
        return;
    }
    t = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    if (t == NULL || t->data == NULL) {
        RING_API_RETNUMBER(0);
        return;
    }
    
    // Returns the actual memory address of the array as a double number
    RING_API_RETNUMBER((double)(size_t)t->data);
}

/* --- INIT --- */
RING_LIBINIT {
    RING_API_REGISTER("tensor_init", ring_tensor_init);
    RING_API_REGISTER("tensor_reshape", ring_tensor_reshape);
    RING_API_REGISTER("tensor_copy", ring_tensor_copy);
    RING_API_REGISTER("tensor_matmul_batch", ring_tensor_matmul_batch); 
    RING_API_REGISTER("tensor_set", ring_tensor_set);
    RING_API_REGISTER("tensor_get", ring_tensor_get);
    
    RING_API_REGISTER("tensor_get_data_ptr", ring_tensor_get_data_ptr);
    
    RING_API_REGISTER("tensor_get_rows", ring_tensor_get_rows);
    RING_API_REGISTER("tensor_get_cols", ring_tensor_get_cols);
    
    RING_API_REGISTER("tensor_add", ring_tensor_add);
    RING_API_REGISTER("tensor_sub", ring_tensor_sub);
    RING_API_REGISTER("tensor_mul_elem", ring_tensor_mul_elem);
    RING_API_REGISTER("tensor_div", ring_tensor_div);
    RING_API_REGISTER("tensor_scalar_mul", ring_tensor_scalar_mul);
    RING_API_REGISTER("tensor_add_scalar", ring_tensor_add_scalar);
    RING_API_REGISTER("tensor_sub_scalar", ring_tensor_sub_scalar);
    
    RING_API_REGISTER("tensor_fill", ring_tensor_fill);
    RING_API_REGISTER("tensor_random", ring_tensor_random);
    RING_API_REGISTER("tensor_square", ring_tensor_square);
    RING_API_REGISTER("tensor_sqrt", ring_tensor_sqrt);
    RING_API_REGISTER("tensor_exp", ring_tensor_exp);
    
    RING_API_REGISTER("tensor_matmul", ring_tensor_matmul);
    RING_API_REGISTER("tensor_transpose", ring_tensor_transpose);
    RING_API_REGISTER("tensor_sum", ring_tensor_sum);
    RING_API_REGISTER("tensor_mean", ring_tensor_mean);
    RING_API_REGISTER("tensor_argmax", ring_tensor_argmax);
    RING_API_REGISTER("tensor_add_row_vec", ring_tensor_add_row_vec);

    RING_API_REGISTER("tensor_sigmoid", ring_tensor_sigmoid);
    RING_API_REGISTER("tensor_sigmoid_prime", ring_tensor_sigmoid_prime);
    RING_API_REGISTER("tensor_tanh", ring_tensor_tanh);
    RING_API_REGISTER("tensor_tanh_prime", ring_tensor_tanh_prime);
    RING_API_REGISTER("tensor_relu", ring_tensor_relu);
    RING_API_REGISTER("tensor_relu_prime", ring_tensor_relu_prime);
    RING_API_REGISTER("tensor_softmax", ring_tensor_softmax);
    
    RING_API_REGISTER("tensor_embedding_forward", ring_tensor_embedding_forward);
    RING_API_REGISTER("tensor_embedding_backward", ring_tensor_embedding_backward);

    RING_API_REGISTER("tensor_layernorm", ring_tensor_layernorm);

    RING_API_REGISTER("tensor_attention_fast", ring_tensor_attention_fast);
    RING_API_REGISTER("tensor_attention_causal", ring_tensor_attention_causal);
    RING_API_REGISTER("tensor_mha_backward_fast", ring_tensor_mha_backward_fast);
    RING_API_REGISTER("tensor_attention_batch", ring_tensor_attention_batch);
    
    RING_API_REGISTER("tensor_select_columns", ring_tensor_select_columns);
    RING_API_REGISTER("tensor_insert_columns", ring_tensor_insert_columns);
    
    
    // --- NEW ---
    RING_API_REGISTER("tensor_slice_rows", ring_tensor_slice_rows);
    RING_API_REGISTER("tensor_insert_rows", ring_tensor_insert_rows);
    
    RING_API_REGISTER("tensor_update_sgd", ring_tensor_update_sgd);
    RING_API_REGISTER("tensor_update_adam", ring_tensor_update_adam);
    RING_API_REGISTER("tensor_dropout", ring_tensor_dropout);

    RING_API_REGISTER("tensor_crossentropy_loss", ring_tensor_crossentropy_loss);
    RING_API_REGISTER("tensor_crossentropy_backward", ring_tensor_crossentropy_backward);

    RING_API_REGISTER("tensor_get_cores", ring_tensor_get_cores);
    RING_API_REGISTER("tensor_set_threads", ring_tensor_set_threads);

    RING_API_REGISTER("tensor_set_from_list", ring_tensor_set_from_list);
    RING_API_REGISTER("tensor_set_one_hot", ring_tensor_set_one_hot);
    
    RING_API_REGISTER("tensor_save", ring_tensor_save);
    RING_API_REGISTER("tensor_load", ring_tensor_load);
    RING_API_REGISTER("tensor_save_fp32", ring_tensor_save_fp32);
    RING_API_REGISTER("tensor_load_fp32", ring_tensor_load_fp32);

    // In-place load functions
    ring_vm_funcregister("tensor_load_inplace", ring_tensor_load_inplace);
    ring_vm_funcregister("tensor_load_fp32_inplace", ring_tensor_load_fp32_inplace);

    RING_API_REGISTER("tensor_clip_global_norm", ring_tensor_clip_global_norm);
    RING_API_REGISTER("tensor_clip_tensor", ring_tensor_clip_tensor);
    
    RING_API_REGISTER("tensor_gelu", ring_tensor_gelu);
    RING_API_REGISTER("tensor_gelu_prime", ring_tensor_gelu_prime);

    RING_API_REGISTER("tensor_sum_squares", ring_tensor_sum_squares);

    RING_API_REGISTER("tensor_repeat_rows", ring_tensor_repeat_rows);

    RING_API_REGISTER("tensor_from_memory", ring_tensor_from_memory);
    

    // --- NEW ---
    RING_API_REGISTER("tensor_attention_linear_causal", ring_tensor_attention_linear_causal);
    RING_API_REGISTER("tensor_attention_linear_optimized", ring_tensor_attention_linear_optimized);
    RING_API_REGISTER("tensor_attention_linear_backward", ring_tensor_attention_linear_backward);
    RING_API_REGISTER("tensor_attention_linear_global_backward", ring_tensor_attention_linear_global_backward);
    
    RING_API_REGISTER("tensor_attention_multihead", ring_tensor_attention_multihead);
    RING_API_REGISTER("tensor_attention_multihead_backward", ring_tensor_attention_multihead_backward);

    RING_API_REGISTER("tensor_set_one_hot_ptr", ring_tensor_set_one_hot_ptr);


    #ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
    #endif
}
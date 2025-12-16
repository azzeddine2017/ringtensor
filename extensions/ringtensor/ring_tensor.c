/*
** RingTensor Extension
** Description: Implementation using Direct Memory Access (Pointers)
** Author: Azzeddine Remmal
*/

#include "ring_tensor.h"

/* --- Memory Management (The Garbage Collector Hook) --- */
void ring_tensor_free(void *pState, void *pPointer) {
    tensor_t *t = (tensor_t *)pPointer;
    if (t != NULL) {
        if (t->data != NULL) {
            free(t->data);
        }
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
    
    // Allocate Struct
    t = (tensor_t *)malloc(sizeof(tensor_t));
    if (!t) { RING_API_ERROR("Malloc Failed (Struct)"); return; }
    
    t->rows = rows;
    t->cols = cols;
    t->size = rows * cols;
    
    // Allocate Data Array (Zero Initialized)
    t->data = (double *)calloc(t->size, sizeof(double));
    if (!t->data) { 
        free(t); 
        RING_API_ERROR("Malloc Failed (Data)"); 
        return; 
    }
    
    // Return Managed Pointer to Ring
    RING_API_RETMANAGEDCPOINTER(t, RING_VM_POINTER_TENSOR, ring_tensor_free);
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
    
    // Safety Check (1-based index from Ring)
    if (r < 1 || r > t->rows || c < 1 || c > t->cols) return;
    
    // Map to 0-based C array
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
/* --- 2. ELEMENT-WISE MATH (Direct Memory Access) -------------------- */
/* ==================================================================== */

void tensor_op_elem(void *pPointer, int op) {
    tensor_t *A, *B;
    int i;

    A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    
    if (A->size != B->size) { RING_API_ERROR("Tensor Size Mismatch"); return; }

    // No Lists, No Lookups, Just Raw Speed
    for(i=0; i<A->size; i++) {
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

/* ==================================================================== */
/* --- 3. SCALARS & TRANSFORMS ---------------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_scalar_mul) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i;
    for(i=0; i<A->size; i++) A->data[i] *= v;
}

RING_FUNC(ring_tensor_add_scalar) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i;
    for(i=0; i<A->size; i++) A->data[i] += v;
}

RING_FUNC(ring_tensor_fill) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double v = RING_API_GETNUMBER(2);
    int i;
    for(i=0; i<A->size; i++) A->data[i] = v;
}

RING_FUNC(ring_tensor_random) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    // srand(time(NULL)); 
    for(i=0; i<A->size; i++) A->data[i] = (double)rand() / (double)RAND_MAX;
}

void tensor_transform(void *pPointer, int op) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    for(i=0; i<A->size; i++) {
        switch(op) {
            case 1: A->data[i] = A->data[i] * A->data[i]; break; // Square
            case 2: A->data[i] = sqrt(A->data[i]); break; // Sqrt
            case 3: A->data[i] = exp(A->data[i]); break; // Exp
        }
    }
}

RING_FUNC(ring_tensor_square) { tensor_transform(pPointer, 1); }
RING_FUNC(ring_tensor_sqrt)   { tensor_transform(pPointer, 2); }
RING_FUNC(ring_tensor_exp)    { tensor_transform(pPointer, 3); }

/* ==================================================================== */
/* --- 4. MATRIX OPS (MATMUL / TRANSPOSE) ----------------------------- */
/* ==================================================================== */

/* 
** Optimized Matrix Multiplication 
** Uses i-k-j loop order for Cache Locality
*/
RING_FUNC(ring_tensor_matmul) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *B = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    tensor_t *C = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    int rA = A->rows;
    int cA = A->cols;
    int cB = B->cols;
    int i, j, k;
    
    if (cA != B->rows) { RING_API_ERROR("MatMul Dims Mismatch"); return; }
    
    // 1. Initialize Result to Zero
    // memset is in <string.h>, but loop is safer if header missing.
    // Optimization: Pointer loop for zeroing
    double *ptrC = C->data;
    int totalElements = rA * cB;
    for(i=0; i<totalElements; i++) *ptrC++ = 0.0;

    // 2. Initialize Pointers to Raw Data
    double *dataA = A->data;
    double *dataB = B->data;
    double *dataC = C->data;

    // 3. Magic Loop (i-k-j) using Pointers
    for(i = 0; i < rA; i++) {
        // Pointer to current row in C
        double *rowC = &dataC[i * cB]; 
        // Pointer to current row in A
        double *rowA = &dataA[i * cA];
        
        for(k = 0; k < cA; k++) {
            // Store value from A once and reuse it
            double valA = rowA[k]; 
            
            // Optimization: Skip if value is 0
            if (valA == 0.0) continue;

            // Pointer to row k in B
            double *rowB = &dataB[k * cB];

            // Inner loop: Only addition and multiplication (no pointer math)
            // This is the form that the CPU loves (SIMD Friendly)
            for(j = 0; j < cB; j++) {
                rowC[j] += valA * rowB[j];
            }
        }
    }
}

/* 
** Optimized Transpose 
** Sequential Write Optimization
*/
RING_FUNC(ring_tensor_transpose) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *C = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    
    int rA = A->rows;
    int cA = A->cols;
    int i, j;
    
    // Iterate over Destination rows (j) then Destination cols (i)
    // This ensures we WRITE to 'C' sequentially, which is faster for the CPU write-buffer
    
    for(j = 0; j < cA; j++) {
        for(i = 0; i < rA; i++) {
            // Dest[j][i] = Src[i][j]
            C->data[j * rA + i] = A->data[i * cA + j];
        }
    }
}

/* ==================================================================== */
/* --- 5. ACTIVATIONS ------------------------------------------------- */
/* ==================================================================== */

void tensor_act(void *pPointer, int op) {
    tensor_t *A = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int i;
    double v;
    for(i=0; i<A->size; i++) {
        v = A->data[i];
        switch(op) {
            case 1: A->data[i] = 1.0 / (1.0 + exp(-v)); break; // Sigmoid
            case 2: A->data[i] = v * (1.0 - v); break;         // SigmoidPrime
            case 3: A->data[i] = tanh(v); break;               // Tanh
            case 4: A->data[i] = 1.0 - (v * v); break;         // TanhPrime
            case 5: A->data[i] = (v > 0) ? v : 0; break;       // ReLU
            case 6: A->data[i] = (v > 0) ? 1.0 : 0.0; break;   // ReLUPrime
        }
    }
}

RING_FUNC(ring_tensor_sigmoid)       { tensor_act(pPointer, 1); }
RING_FUNC(ring_tensor_sigmoid_prime) { tensor_act(pPointer, 2); }
RING_FUNC(ring_tensor_tanh)          { tensor_act(pPointer, 3); }
RING_FUNC(ring_tensor_tanh_prime)    { tensor_act(pPointer, 4); }
RING_FUNC(ring_tensor_relu)          { tensor_act(pPointer, 5); }
RING_FUNC(ring_tensor_relu_prime)    { tensor_act(pPointer, 6); }

RING_FUNC(ring_tensor_softmax) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int r, c;
    for(r=0; r<T->rows; r++) {
        double maxVal = -DBL_MAX;
        int rowOff = r * T->cols;
        // Find Max
        for(c=0; c<T->cols; c++) if(T->data[rowOff+c] > maxVal) maxVal = T->data[rowOff+c];
        
        // Exp & Sum
        double sum = 0.0;
        for(c=0; c<T->cols; c++) {
            T->data[rowOff+c] = exp(T->data[rowOff+c] - maxVal);
            sum += T->data[rowOff+c];
        }
        // Normalize
        for(c=0; c<T->cols; c++) {
            if(sum!=0) T->data[rowOff+c] /= sum;
        }
    }
}

/* ==================================================================== */
/* --- 6. UTILITIES & OPTIMIZERS -------------------------------------- */
/* ==================================================================== */

RING_FUNC(ring_tensor_sum) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    int axis = (int)RING_API_GETNUMBER(2);
    tensor_t *R = (tensor_t *)RING_API_GETCPOINTER(3, RING_VM_POINTER_TENSOR);
    
    if (axis == 1) { // Sum Rows -> Result (Rows x 1)
        for(int r=0; r<T->rows; r++) {
            double s = 0;
            for(int c=0; c<T->cols; c++) s += T->data[r*T->cols + c];
            R->data[r] = s;
        }
    } else { // Sum Cols -> Result (1 x Cols)
        // Reset Result first
        for(int i=0; i<R->size; i++) R->data[i] = 0.0;
        
        for(int r=0; r<T->rows; r++) {
            for(int c=0; c<T->cols; c++) {
                R->data[c] += T->data[r*T->cols + c];
            }
        }
    }
}

RING_FUNC(ring_tensor_mean) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double sum = 0;
    int i;
    for(i=0; i<T->size; i++) sum += T->data[i];
    RING_API_RETNUMBER(sum / T->size);
}

RING_FUNC(ring_tensor_argmax) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *R = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    
    for(int r=0; r<T->rows; r++) {
        double maxVal = -DBL_MAX;
        int maxIdx = 1;
        for(int c=0; c<T->cols; c++) {
            if(T->data[r*T->cols + c] > maxVal) {
                maxVal = T->data[r*T->cols + c];
                maxIdx = c + 1; // 1-based index for Ring
            }
        }
        R->data[r] = (double)maxIdx;
    }
}

RING_FUNC(ring_tensor_dropout) {
    tensor_t *T = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    double rate = RING_API_GETNUMBER(2);
    double scale = 1.0 / (1.0 - rate);
    int i;
    for(i=0; i<T->size; i++) {
        double rnd = (double)rand() / (double)RAND_MAX;
        if(rnd < rate) T->data[i] = 0.0;
        else T->data[i] *= scale;
    }
}

RING_FUNC(ring_tensor_update_sgd) {
    tensor_t *W = (tensor_t *)RING_API_GETCPOINTER(1, RING_VM_POINTER_TENSOR);
    tensor_t *G = (tensor_t *)RING_API_GETCPOINTER(2, RING_VM_POINTER_TENSOR);
    double lr = RING_API_GETNUMBER(3);
    int i;
    for(i=0; i<W->size; i++) W->data[i] -= (lr * G->data[i]);
}

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
    
    double corr1 = 1.0 - pow(b1, t);
    double corr2 = 1.0 - pow(b2, t);
    if(corr1 == 0) corr1 = 1e-9;
    if(corr2 == 0) corr2 = 1e-9;
    
    int i;
    for(i=0; i<W->size; i++) {
        double g = G->data[i];
        
        M->data[i] = (b1 * M->data[i]) + ((1.0 - b1) * g);
        V->data[i] = (b2 * V->data[i]) + ((1.0 - b2) * g * g);
        
        double m_hat = M->data[i] / corr1;
        double v_hat = V->data[i] / corr2;
        if(v_hat < 0) v_hat = 0;
        
        W->data[i] -= (lr * m_hat) / (sqrt(v_hat) + eps);
    }
}

/* --- INIT --- */
RING_LIBINIT {
    RING_API_REGISTER("tensor_init", ring_tensor_init);
    RING_API_REGISTER("tensor_set", ring_tensor_set);
    RING_API_REGISTER("tensor_get", ring_tensor_get);
    
    RING_API_REGISTER("tensor_add", ring_tensor_add);
    RING_API_REGISTER("tensor_sub", ring_tensor_sub);
    RING_API_REGISTER("tensor_mul_elem", ring_tensor_mul_elem);
    RING_API_REGISTER("tensor_div", ring_tensor_div);
    RING_API_REGISTER("tensor_scalar_mul", ring_tensor_scalar_mul);
    RING_API_REGISTER("tensor_add_scalar", ring_tensor_add_scalar);
    
    RING_API_REGISTER("tensor_fill", ring_tensor_fill);
    RING_API_REGISTER("tensor_random", ring_tensor_random);
    RING_API_REGISTER("tensor_square", ring_tensor_square);
    RING_API_REGISTER("tensor_sqrt", ring_tensor_sqrt);
    RING_API_REGISTER("tensor_exp", ring_tensor_exp);
    
    RING_API_REGISTER("tensor_matmul", ring_tensor_matmul);
    RING_API_REGISTER("tensor_transpose", ring_tensor_transpose);
    RING_API_REGISTER("tensor_sum", ring_tensor_sum);
    RING_API_REGISTER("tensor_mean", ring_tensor_mean);
    
    RING_API_REGISTER("tensor_sigmoid", ring_tensor_sigmoid);
    RING_API_REGISTER("tensor_sigmoid_prime", ring_tensor_sigmoid_prime);
    RING_API_REGISTER("tensor_tanh", ring_tensor_tanh);
    RING_API_REGISTER("tensor_tanh_prime", ring_tensor_tanh_prime);
    RING_API_REGISTER("tensor_relu", ring_tensor_relu);
    RING_API_REGISTER("tensor_relu_prime", ring_tensor_relu_prime);
    RING_API_REGISTER("tensor_softmax", ring_tensor_softmax);
    
    RING_API_REGISTER("tensor_update_sgd", ring_tensor_update_sgd);
    RING_API_REGISTER("tensor_update_adam", ring_tensor_update_adam);
    RING_API_REGISTER("tensor_dropout", ring_tensor_dropout);
    RING_API_REGISTER("tensor_argmax", ring_tensor_argmax);
}
# The Main File
load "stdlib.ring"


func main
	? "
  _____  _             _______
 |  __ \(_)           |__   __|
 | |__) |_ _ __   __ _   | | ___ _ __  ___  ___  _ __
 |  _  /| | '_ \ / _` |  | |/ _ \ '_ \/ __|/ _ \| '__|
 | | \ \| | | | | (_| |  | |  __/ | | \__ \ (_) | |
 |_|  \_\_|_| |_|\__, |  |_|\___|_| |_|___/\___/|_|
                  __/ |
                 |___/
    "
	? "Welcome to RingTensor v1.0.0"
	? "High-Performance C Extension for Deep Learning in Ring"
	? copy("=", 80)

	? "### 1. Lifecycle & Access"
	? "--------------------------------------------------------------------------------"
	? "Function         Parameters          Return   Description"
	? "--------------------------------------------------------------------------------"
	? "tensor_init      Rows, Cols          Pointer  Allocates memory (init to 0.0)"
	? "tensor_set       Ptr, R, C, Val      -        Sets value at (Row, Col)"
	? "tensor_get       Ptr, R, C           Number   Gets value from (Row, Col)"
	? ""
	? copy("=", 80)
	? "### 2. Element-Wise Math (In-Place) - Modifies First Tensor (A)"
	? "--------------------------------------------------------------------------------"
	? "Function           Parameters          Logic"
	? "--------------------------------------------------------------------------------"
	? "tensor_add         Ptr A, Ptr B        A += B"
	? "tensor_sub         Ptr A, Ptr B        A -= B"
	? "tensor_mul_elem    Ptr A, Ptr B        A *= B (Hadamard)"
	? "tensor_div         Ptr A, Ptr B        A /= B"
	? "tensor_scalar_mul  Ptr A, Number n     A *= n"
	? "tensor_add_scalar  Ptr A, Number n     A += n"
	? ""
	? copy("=", 80)
	? "### 3. Matrix Operations"
	? "--------------------------------------------------------------------------------"
	? "Function          Parameters             Description             Behavior"
	? "--------------------------------------------------------------------------------"
	? "tensor_matmul     Ptr A, Ptr B, Ptr Res  Dot Product (A x B)     Writes to Res"
	? "tensor_transpose  Ptr A, Ptr Res         Transposes A            Writes to Res"
	? "tensor_sum        Ptr A, Axis, Ptr Res   1=Rows, 0=Cols          Writes to Res"
	? "tensor_mean       Ptr A                  Mean of all items       Returns Number"
	? "tensor_argmax     Ptr A, Ptr Res         Max index per row       Writes to Res"
	? ""
	? copy("=", 80)
	? "### 4. Transformations & Activations (In-Place)"
	? "--------------------------------------------------------------------------------"
	? "Function          Formula / Description"
	? "--------------------------------------------------------------------------------"
	? "tensor_fill       Fills with value n"
	? "tensor_random     Fills with 0.0 to 1.0"
	? "tensor_square     x^2"
	? "tensor_sqrt       sqrt(x)"
	? "tensor_exp        e^x"
	? "tensor_sigmoid    1 / (1 + e^-x)"
	? "tensor_tanh       tanh(x)"
	? "tensor_relu       max(0, x)"
	? "tensor_softmax    Stable Softmax (Exp-Normalize)"
	? ""
	? copy("=", 80)
	? "### 5. Optimizers (Fused Kernels)"
	? "High-performance updates that happen entirely in C."
	? "--------------------------------------------------------------------------------"
	? "tensor_update_sgd(Ptr W, Ptr Grad, Number LR)"
	? "tensor_update_adam(Ptr W, Ptr G, Ptr M, Ptr V, LR, Beta1, Beta2, Eps, T)"
	? "tensor_dropout(Ptr A, Number Rate)"
	? ""
	? copy("=", 80)
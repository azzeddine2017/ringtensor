#include "ring_tensor.h"


if iswindows()
	LoadLib("ring_tensor.dll")
but ismacosx()
	LoadLib("libring_tensor.dylib")
else
	LoadLib("libring_tensor.so")
ok


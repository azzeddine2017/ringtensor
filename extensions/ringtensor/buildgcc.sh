gcc -c -fpic -O2 ring_tensor.c -I $PWD/../../language/include 
gcc -shared -o $PWD/../../lib/libring_tensor.so ring_tensor.o -L $PWD/../../lib -lring

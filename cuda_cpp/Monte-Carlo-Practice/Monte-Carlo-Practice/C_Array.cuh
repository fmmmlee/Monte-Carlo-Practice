#ifndef _C_ARR_
#define _C_ARR_

#include <cuda_runtime.h>

/* atomicCAS is the key operation here */
/* also create a separate .h file with functions for host to create a C_Array on a device */
//TODO: create device error handler
//TODO: Add exclusive or whatever it's called

//currently only supports objects of size 32 or 64 bits
template <class T>
class C_Array
{
	T* start_ptr;

public:
	C_Array(int quantity)
	{
		/* allocates enough space for quantity objects of the declared type */
		cudaMalloc((void**)&start_ptr[0], quantity*sizeof(T));
	}

	~C_Array()
	{
		cudaFree(start_ptr);
	}

	T* get(int i)
	{
		return &start_ptr[i];
	}

	void set(T* val, int i)
	{
		atomicExch(&start[i], val);
	}





}

#endif
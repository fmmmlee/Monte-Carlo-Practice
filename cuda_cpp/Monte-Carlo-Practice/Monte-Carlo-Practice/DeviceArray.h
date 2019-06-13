#ifndef _DEVICE_ARRAY_
#define _DEVICE_ARRAY_


#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include "cudaErrHandler.h"

/* based on https://www.quantstart.com/articles/dev_array_A_Useful_Array_Class_for_CUDA */
using namespace std;

template <class T>

class DeviceArray
{
public:

	/* constructor */
	DeviceArray(int size)
	{
		cudaErrChk(cudaMalloc((void**)&start_ptr, sizeof(T)*size));
		this->size = size;
	}

	/* destructor */
	~DeviceArray()
	{
		this->free(); //'this' here for clarity
	}
	
	/* resize */
	void resize(int size)
	{
		free();
		cudaErrChk(cudaMalloc((void**)&start_ptr, sizeof(T)*size));
		this->size = size;
	}


	/*********** GETTERS ***********/

	/* pointer to single value */
	T* get(int i)
	{
		return start_ptr[i];
	}

	/* pointer to whole array */
	T* get()
	{
		return start_ptr;
	}
	
	/* group of values of size quantity starting at index i */
	/* if i + quantity reaches outside the bounds of allocated memory, will return values up to the end of the array instead */
	void get(T* destination, int quantity, int i)
	{
		int reqSize = std::min(quantity+i, size);
		cudaErrChk(cudaMemcpy(destination, &start_ptr[i], sizeof(T)*reqSize, cudaMemcpyDeviceToHost));
	}

	/* get/copier - group of values of size quantity (or size of the array, whichever is smaller) starting at beginning of device array */
	void get(T* destination, int quantity)
	{
		int reqSize = std::min(quantity, size);
		cudaErrChk(cudaMemcpy(destination, start_ptr, sizeof(T)*reqSize, cudaMemcpyDeviceToHost));
	}
	
	int length()
	{
		return size;
	}

	/*********** SETTERS ***********/

	/* quantity many values pointed to by src */
	void set(const T* src, int quantity)
	{
		if (quantity > size)
			throw std::runtime_error("specified quantity is larger than the destination device array");
		else
			cudaErrChk(cudaMemcpy(start_ptr, src, sizeof(T)*count, cudaMemcpyHostToDevice));
	}

private:

	/* small function for housekeeping */
	void free()
	{
		cudaErrChk(cudaFree(start_ptr));
		size = 0;
		start_ptr = 0;
	}

	/* variables */
	T* start_ptr;
	int size;
};
#endif
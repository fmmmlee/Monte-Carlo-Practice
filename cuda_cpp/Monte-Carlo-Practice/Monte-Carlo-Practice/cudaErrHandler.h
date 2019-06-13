#ifndef __CUDA_ERR_CHK__
#define __CUDA_ERR_CHK__

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaErrChk(code) {cudaCodeChk(code, __FILE__, __LINE__);}

/* inspired by talonmies's great stackexchange answer here: https://stackoverflow.com/questions/14038589/ */
inline void cudaCodeChk(cudaError_t code, const char* file, int line, bool abort=1)
{
	if(code != cudaSuccess)
	{
		fprintf(stderr, "Error in %s line %d, CUDA API error: %s", file, line, cudaGetErrorString(code));
		if (abort)
			exit(code);
	}
}
#endif


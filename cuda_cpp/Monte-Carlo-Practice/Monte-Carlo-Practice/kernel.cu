/*
	(c) Matthew Lee
	Spring 2019
	MIT License
*/


#include <stdio.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "barrier_options.cuh"

void down_out(unsigned N_STEPS, unsigned N_PATHS, float start_price, float sigma, float mu, float time, float barrier_u, float strike_price, float* price_est, float* payoff);
__global__ void barrier_simulation(unsigned total_paths, unsigned steps_per_sim, float start_price, float sigma, float mu, float time, float barrier_a, float strike_price, float* randoms, float* price_est, float* payoff);

__global__ void barrier_simulation(
	unsigned total_paths,
	unsigned steps_per_sim,
	float start_price,
	float sigma,
	float mu,
	float time,
	float barrier_a,
	float strike_price,
	float* randoms,
	float* price_est,
	float* payoff)
{
	
	//index of destination for simulated price and payoff
	//TODO: Add block/grid index etc
	const int result_index = blockIdx.x*blockDim.x + threadIdx.x;

	//only continue execution if not finished with job
	if (result_index > total_paths)
		return;
	
	//initializing the starting price of the option
	float price = start_price;

	//barrier adjustment
	float barrier = barrier_a * (exp(+0.5826*sigma*sqrt(time / steps_per_sim)));

	//time per iteration
	float dt = (float)time / steps_per_sim;

	//
	float v = mu - ((sigma*sigma) / 2);
	
	for (int i = 0; i < steps_per_sim; i++)
	{
		//discretized stoichastic difeq (geometric brownian motion), apparently - must investigate the math behind this further
		price = price * exp(v*dt + sigma*randoms[result_index+i]);
	}

	__syncthreads();
	//after completing specified number of steps, input price_est into array
	price_est[result_index] = (price >= barrier ? price : 0.0);

	//input expected payoff into array
	if(price >= barrier)
	{
		float thispayoff = price - strike_price;
		//risk = mu calculation
		payoff[result_index] = (thispayoff > 0.0 ? thispayoff * exp((-mu)*time) : 0.0);
	}
	else {
		payoff[result_index] = 0.0;
	}
	__syncthreads();
}
 
void down_out(unsigned N_STEPS, unsigned N_PATHS, float start_price, float sigma, float mu, float time, float barrier_u, float strike_price, float* price_est, float* payoff)
{
	const unsigned BLOCK_SIZE = 1024;
	const unsigned GRID_SIZE = ceil(float(N_PATHS) / float(BLOCK_SIZE));
	//generating random floats on device memory
	float* randoms;
	curandGenerator_t gen;
	cudaMalloc(&randoms, N_STEPS*N_PATHS*sizeof(float));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, clock());
	curandGenerateNormal(gen, randoms, N_STEPS*N_PATHS, 0, sqrt((float)time/N_STEPS));
	
	//calling simulation kernel
	barrier_simulation <<<GRID_SIZE, BLOCK_SIZE>>> (N_PATHS, N_STEPS, start_price, sigma, mu, time, barrier_u, strike_price, randoms, price_est, payoff);
}
#ifndef _KERNEL_HEADER_
#define _KERNEL_HEADER_

void down_out(unsigned N_STEPS, unsigned N_PATHS, float start_price, float sigma, float mu, float time, float barrier_u, float strike_price, float* price_est, float* payoff);
void __syncthreads();
#endif

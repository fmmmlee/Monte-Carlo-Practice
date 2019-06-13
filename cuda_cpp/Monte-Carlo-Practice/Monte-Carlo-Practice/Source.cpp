#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include "barrier_options.cuh"
#include "DeviceArray.h"

using namespace std;

/*
 * Matthew Lee
 * Spring 2019
 * 
 *
 */


//TODO: Change kernel to use Kahan summation
//TODO: Add option to write to file/database for metrics, with codes like [CLOCKS] and [TOTAL] for easier parsing later
//TODO: Graphing utility
//TODO: use GMP for GPU accelerated array sum
//TODO: Calculate total error from ideal based on floating point error in single and float precision floats, number of iterations, number of path divergences per iteration, etc.
//full explanation of estimating down-and-out exotic option prices can be found at https://pdfs.semanticscholar.org/542f/6e1e9338632e3bc5b56dad2515854e34190f.pdf

int main()
{
	try
	{
		/* variables used in kernel args and run configuration */
		const size_t TOT_SIMS = 3200000;
		const size_t PER_SIM = 365;
		const float mu = 0.1f;
		const float sigma = 0.2f;
		const float years = 1.0f;
		const float start_price = 100.0f;
		const float strike_price = 100.0f;
		const float barrier = 95.0f;
		vector<float> price_est(TOT_SIMS);
		vector<float> payoff(TOT_SIMS);
		DeviceArray<float> price_est_dev(TOT_SIMS);
		DeviceArray<float> payoff_dev(TOT_SIMS);


		cout << "======================================================\n";
		cout << "======================================================\n";
		
		double start_time = double(clock()) / CLOCKS_PER_SEC;

		down_out(PER_SIM, TOT_SIMS, start_price, sigma, mu, years, barrier, strike_price, price_est_dev.get(), payoff_dev.get());
		cudaDeviceSynchronize();

		price_est_dev.get(&price_est[0], TOT_SIMS);
		payoff_dev.get(&payoff[0], TOT_SIMS);

		double end_time = double(clock()) / CLOCKS_PER_SEC;

		double temp_sum = 0.0;
		for (size_t i = 0; i < TOT_SIMS; i++)
		{
			temp_sum += payoff[i];
		}
		temp_sum /= TOT_SIMS;

		double temp_sum2 = 0.0;
		for (size_t i = 0; i < TOT_SIMS; i++)
		{
			temp_sum2 += price_est[i];
		}
		temp_sum2 /= TOT_SIMS;

		cout<<"[STATUS] Kernel and Cleanup Finished\n";
		cout << "===================================================\n";
		cout<<"[GPU] Length of each simulation: " << years << " years\n";
		cout<<"[GPU] Price path decisions calculated at " << PER_SIM << " intervals.\n";
		cout<<"======================================================\n";
		cout<<"[GPU] Number of individual simulations run: " << TOT_SIMS << "\n";
		cout<<"[GPU] Kernel Execution Time: " << (end_time - start_time)*1e3 << " ms\n";
		cout<<"======================================================\n";
		cout<<"[GPU] Projected option return: "<< temp_sum << "\n";
		cout << "[GPU] Projected option price: " << temp_sum2 << "\n";
		
		/*for (int i = 0; i < 100; i++)
		{
			cout<<"payoff value " << i << ": " << payoff[i] << "\n";
			cout<< "estimated option value " << i << ": " << price_est[i] << "\n";
		}*/

		/*
		cout<<"[GPU] Average projected option price after the time period specified: " << res << " (average calculated using concurrent function)");
		cout<<"[GPU] Time to compute average:" << Time.from_nano(end_mult - start_mult));
		cout<<"[GPU] Projected option payoff after the time period specified: " << pay << " (average calculated using concurrent function)");
		cout<<"======================================================");
		cout<<"======================================================");
		*/
	}
	catch (exception& e) {
		cout << "exception: " << e.what() << "\n\n";
	}
	
}

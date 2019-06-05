package barrier_options;

import java.math.BigDecimal;

import bundled_utilities.Average_of_Array;
import bundled_utilities.Time;

/*
 * 
 * Matthew Lee
 * Spring 2019
 * Practicing Monte Carlo Simulation
 * 
 * 
 * 
 * Using double to see how much faster it is
 * 
 */
public class CPU_approx {

	
	/*************MULTITHREADED*************/
	
	public static void threaded(int iterations, OptionParams requested) throws InterruptedException
	{		
		/* padding */
		System.out.println("======================================================");
		System.out.println("======================================================");
		
		int max_threads = Runtime.getRuntime().availableProcessors();
		
		/* printing some basic thread info */

		System.out.println("[CPU] Thread count = " + max_threads);
		
		int per_thread = iterations/max_threads;
		int extras = iterations-(per_thread*max_threads);
		
		int size = iterations;
		double result[] = new double[size];

		final double mu = requested.mu;
		final double sigma = requested.sigma;
		final double time = requested.years;
		final double start_price = requested.start_price;
		final int steps_per_sim = requested.steps_per_sim;
		final double barrier = requested.barrier;
		
		long start_time = System.nanoTime();
		
		/* spinning threads */
		Thread[] threads = new Thread[max_threads];
		for(int i = 0; i < max_threads; i++)
			threads[i] = new Thread(new CPU_Thread(result, steps_per_sim, start_price, sigma, mu, time, i, per_thread, barrier));
		Thread remainder = new Thread(new CPU_Thread(result, steps_per_sim, start_price, sigma, mu, time, max_threads, extras, barrier));
		
		/* starting threads */
		for(int i = 0; i < max_threads; i++)
			threads[i].start();
		remainder.start();
		
		/* waiting on all threads */
		for(int i = 0; i < max_threads; i++)
			threads[i].join();
		remainder.join();
		
		/* getting result and printing stats/output to console */
		long end_time = System.nanoTime();
		
		long start_avg_time = System.nanoTime();
		BigDecimal average =  Average_of_Array.avg_double(result);
		long end_avg_time = System.nanoTime();
		
		long total_time = end_time - start_time;
		long total_avg_time = end_avg_time - start_avg_time;
		System.out.println("[CPU] Number of individual simulations run: " + iterations);
		System.out.println("[CPU] Average time per calculation:" + Time.from_nano(total_time/iterations));
		System.out.println("[CPU] Total execution time:" + Time.from_nano(total_time));
		System.out.println("[CPU] Projected option price after the time period specified: " + average);
		System.out.println("[CPU] Time to compute average:" + Time.from_nano(total_avg_time));
		
		/* padding */
		System.out.println("======================================================");
		System.out.println("======================================================");
	}
}

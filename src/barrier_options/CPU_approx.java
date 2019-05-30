package barrier_options;

/*
 * 
 * Matthew Lee
 * Spring 2019
 * Practicing Monte Carlo Simulation
 * 
 * 
 * 
 * Using float to see how much faster it is
 * 
 */
public class CPU_approx {

	
	/*************MULTITHREADED*************/
	
	public static void threaded(int iterations) throws InterruptedException
	{		

		int max_threads = Runtime.getRuntime().availableProcessors();

		/* printing some basic thread info */

		System.out.println("[CPU] Thread count = " + max_threads);
		
		int per_thread = iterations/max_threads;
		int extras = iterations-(per_thread*max_threads);
		
		int size = iterations;
		float result[] = new float[size];

		final float mu = 0.1f;
		final float sigma = 0.1f;
		final float time = 1.0f;
		final float start_price = 100.0f;
		final int steps_per_sim = 365;
		
		long start_time = System.nanoTime();
		
		/* spinning threads */
		Thread[] threads = new Thread[max_threads];
		for(int i = 0; i < max_threads; i++)
			threads[i] = new Thread(new CPU_Thread(result, steps_per_sim, start_price, sigma, mu, time, i, per_thread));
		Thread remainder = new Thread(new CPU_Thread(result, steps_per_sim, start_price, sigma, mu, time, 12, extras));
		
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
		long huge = 0;
		int realsize = 0;
		for(float a : result)
		{
			if(a != 0)
			{
				realsize = realsize+1;
				huge+=(long)a;
			}
		}
		
		long total_time = end_time - start_time;
		System.out.println("[CPU] Number of individual simulations run: " + iterations);
		System.out.println("[CPU] Average time per calculation:" + bundled_utilities.Time.from_nano(total_time/iterations));
		System.out.println("[CPU] Total execution time:" + bundled_utilities.Time.from_nano(total_time));
		System.out.println("[CPU] Projected option price after the time period specified: " + (float)huge/realsize);
	}
}

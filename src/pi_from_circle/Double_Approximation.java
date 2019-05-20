package pi_from_circle;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;

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
public class Double_Approximation {

	
	
	/*************SINGLE THREAD*************/
	public void sequential(long iterations)
	{
		long points_in_circle = 0;
		long start_time = System.nanoTime();
		/* looping */
		for(int i = 0; i <= iterations; i++)
		{
			Double x = ThreadLocalRandom.current().nextDouble();
			Double y = ThreadLocalRandom.current().nextDouble();
			
			if((x*x)+(y*y) <= 1)	//checks if value is within the section, based on the formula for an ellipse
				points_in_circle++;
		}
		Printing_Results.console("Single Threaded Double",points_in_circle, iterations, System.nanoTime()-start_time);
	}	


	
	
	/*************MULTITHREADED*************/
	
	public static void threaded(long iterations) throws InterruptedException
	{		
		long start_time = System.nanoTime();
		
		AtomicLong atomic_result = new AtomicLong(0);

		int max_threads = Runtime.getRuntime().availableProcessors();

		long per_thread = iterations/max_threads;				//how many iterations each thread needs to do
		long extra_runs = iterations - per_thread*max_threads;	//this is for when the jobs didn't divide cleanly
		
		/* printing some basic thread info */
		System.out.println("Thread count = " + max_threads);
		System.out.println("Points per thread: " + per_thread);
		System.out.println("Points for remainder thread: " + extra_runs); 
		
		/* spinning threads */
		Thread[] threads = new Thread[max_threads];
		for(int i = 0; i < max_threads; i++)
			threads[i] = new Thread(new Double_Thread(per_thread, atomic_result));
		Thread remainder = new Thread(new Double_Thread(extra_runs, atomic_result));
		
		/* starting threads */
		for(int i = 0; i < max_threads; i++)
			threads[i].start();
		remainder.start();
		
		/* waiting on all threads */
		for(int i = 0; i < max_threads; i++)
			threads[i].join();
		remainder.join();
		
		/* getting result and printing stats/output to console */
		long result = atomic_result.get();
		Printing_Results.console("Multithreaded Double",result, iterations, System.nanoTime()-start_time);
	}
}

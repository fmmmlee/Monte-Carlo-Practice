package barrier_options;

import java.util.Random;
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
public class CPU_approx {

	
	/*************MULTITHREADED*************/
	
	public static void threaded(int iterations) throws InterruptedException
	{		
		long start_time = System.nanoTime();

		int max_threads = Runtime.getRuntime().availableProcessors();

		/* printing some basic thread info */
		System.out.println("========================================");
		System.out.println("Thread count = " + max_threads);
		
		int per_thread = iterations/max_threads;
		System.out.println(per_thread);
		int extras = iterations-(per_thread*max_threads);
		
		int size = iterations;
		double array_one[] = new double[size];
		double array_two[] = new double[size];
		double result[] = new double[size];
		
		Random anewrandom = new Random();
		
		for(int i = 0; i < size; i++)
		{
			array_one[i] = anewrandom.nextDouble();
			array_two[i] = anewrandom.nextDouble();
		}
		
		/* spinning threads */
		Thread[] threads = new Thread[max_threads];
		for(int i = 0; i < max_threads; i++)
			threads[i] = new Thread(new CPU_Thread(array_one, array_two, result, i, per_thread));
		Thread remainder = new Thread(new CPU_Thread(array_one, array_two, result, 12, extras));
		
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
		System.out.println(bundled_utilities.Time.from_nano(end_time-start_time));
	}
}

package pi_from_circle;

import java.math.BigDecimal;
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
 * Using BigDecimal to increase the accuracy of the decision if a point is in the circle or not
 * 
 */
public class BigDecimal_Approximation {
	
	
	
	/*************SINGLE THREAD*************/
	public void sequential(long iterations)
	{
		final BigDecimal max = new BigDecimal(1.0);
		final BigDecimal min = new BigDecimal(0.0);

		final BigDecimal one = new BigDecimal(1.0);
		
		int in_circle = 0;
		long start_time = System.nanoTime();
		/* looping */
		for(int i = 0; i <= iterations; i++)
		{
			BigDecimal x = randomBigDecimal(min, max);
			BigDecimal y = randomBigDecimal(min, max);
			
			if((x.multiply(x)).add((y.multiply(y))).compareTo(one) <= 0)
				in_circle++;
		}
		long time = System.nanoTime() - start_time;
		Printing_Results.console("Single Thread BigDecimal", in_circle, iterations, time);
	}
	
	
	
	
	
	
	/*************MULTITHREADED*************/
	public void threaded(long iterations) throws InterruptedException
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
			threads[i] = new Thread(new BigDecimal_Thread(per_thread, atomic_result));
		Thread remainder = new Thread(new BigDecimal_Thread(extra_runs, atomic_result));
		
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
		Printing_Results.console("Multithreaded BigDecimal", result, iterations, System.nanoTime()-start_time);
	}
	
	
	/*************extras*************/
	
	/* make a new decimal within the range of min to max */
	public static BigDecimal randomBigDecimal(BigDecimal min, BigDecimal max)
	{
        BigDecimal randFromDouble = new BigDecimal(ThreadLocalRandom.current().nextDouble());
        BigDecimal actualRandomDec = randFromDouble.divide(max,BigDecimal.ROUND_DOWN);
        return min.add(actualRandomDec);
	}
	
	
}

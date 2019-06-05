package bundled_utilities;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;

/*
 * 
 * Matthew Lee
 * Spring 2019
 * 
 * 
 */

//TODO: Unit tests

public class Average_of_Array
{
	
	/*********DOUBLE**********/
	public static BigDecimal avg_double(double[] input) throws InterruptedException
	{
		BigDecimal result = new BigDecimal(0.0f);
		int max_threads = Runtime.getRuntime().availableProcessors();
		int iterations = input.length;
		int per_thread = iterations/max_threads;
		int extras = iterations-(per_thread*max_threads);
		//values to add
		double[][] thread_args = new double[max_threads+1][per_thread];
		//destination of individual computations
		BigDecimal_In_Thread[] averages = new BigDecimal_In_Thread[max_threads+1];
		for(int i = 0; i < averages.length; i++)
		{
			averages[i] = new BigDecimal_In_Thread();
			averages[i].just = BigDecimal.valueOf(0.0f);
			if(i < max_threads)
				averages[i].length = per_thread;
			else
				averages[i].length = extras;
		}
		
		/* setting the thread arguments */
		for(int i = 0; i < max_threads; i++)
		{
				thread_args[i] = Arrays.copyOfRange(input, per_thread*i, (per_thread*i) + per_thread);
		}		
		thread_args[max_threads] = Arrays.copyOfRange(input, per_thread*max_threads, (per_thread*max_threads) + extras);
		
		/* spinning threads */
		Thread[] threads = new Thread[max_threads];
		for(int i = 0; i < max_threads; i++)
			threads[i] = new Thread(new Double_Thread(thread_args[i], averages[i]));
		Thread remainder = new Thread(new Double_Thread(thread_args[max_threads], averages[max_threads]));
		
		/* starting threads */
		for(int i = 0; i < max_threads; i++)
			threads[i].start();
		remainder.start();
		
		/* waiting on all threads */
		for(int i = 0; i < max_threads; i++)
			threads[i].join();
		remainder.join();
		
		int total_total = 0;
		for(int i = 0; i < averages.length; i++)
		{
			result = result.add(averages[i].just);
			total_total += averages[i].length;
		}
		
		return result.divide(BigDecimal.valueOf(total_total), RoundingMode.HALF_UP);
	}

	/*********FLOAT**********/
	
	public static BigDecimal avg_float(float[] input) throws InterruptedException
	{
		BigDecimal result = new BigDecimal(0.0f);
		int max_threads = Runtime.getRuntime().availableProcessors();
		int iterations = input.length;
		int per_thread = iterations/max_threads;
		int extras = iterations-(per_thread*max_threads);
		//values to add
		float[][] thread_args = new float[max_threads+1][per_thread];
		//destination of individual computations
		BigDecimal_In_Thread[] averages = new BigDecimal_In_Thread[max_threads+1];
		for(int i = 0; i < averages.length; i++)
		{
			averages[i] = new BigDecimal_In_Thread();
			averages[i].just = BigDecimal.valueOf(0.0f);
			if(i < max_threads)
				averages[i].length = per_thread;
			else
				averages[i].length = extras;
		}
		
		/* setting the thread arguments */
		for(int i = 0; i < max_threads; i++)
		{
				thread_args[i] = Arrays.copyOfRange(input, per_thread*i, (per_thread*i) + per_thread);
		}		
		thread_args[max_threads] = Arrays.copyOfRange(input, per_thread*max_threads, (per_thread*max_threads) + extras);
		
		/* spinning threads */
		Thread[] threads = new Thread[max_threads];
		for(int i = 0; i < max_threads; i++)
			threads[i] = new Thread(new Float_Thread(thread_args[i], averages[i]));
		Thread remainder = new Thread(new Float_Thread(thread_args[max_threads], averages[max_threads]));
		
		/* starting threads */
		for(int i = 0; i < max_threads; i++)
			threads[i].start();
		remainder.start();
		
		/* waiting on all threads */
		for(int i = 0; i < max_threads; i++)
			threads[i].join();
		remainder.join();
		
		int total_total = 0;
		for(int i = 0; i < averages.length; i++)
		{
			result = result.add(averages[i].just);
			total_total += averages[i].length;
		}
		
		return result.divide(BigDecimal.valueOf(total_total), RoundingMode.HALF_UP);
	}
}

class BigDecimal_In_Thread
{
	BigDecimal just;
	int length;
}

class Double_Thread implements Runnable
{
	final double average_this[];
	int size;
	BigDecimal_In_Thread total;
	
	Double_Thread(double[] work, BigDecimal_In_Thread total)
	{
		average_this = work;
		size = average_this.length;
		this.total = total;
	}
	
	public void run()
	{
		for(double item : average_this)
		{
			if(!Double.isNaN(item) && !Double.isInfinite(item))
			{
				try
				{
					total.just = total.just.add(BigDecimal.valueOf(item));
				} catch(NumberFormatException e) {
					System.err.println("Invalid double is: " + item);
					e.printStackTrace();
					
				}
				
			} else {
				total.length -= 1; //adjusting number to divide sum if we skip a NaN entry (to ensure average ignores NaNs)
			}
			
		}
	}	
}

class Float_Thread implements Runnable
{
	final float average_this[];
	int size;
	BigDecimal_In_Thread total;
	
	Float_Thread(float[] work, BigDecimal_In_Thread total)
	{
		average_this = work;
		size = average_this.length;
		this.total = total;
	}
	
	public void run()
	{
		for(float item : average_this)
		{
			if(!Float.isNaN(item) && !Float.isInfinite(item))
			{
				try
				{
					total.just = total.just.add(BigDecimal.valueOf(item));
				} catch(NumberFormatException e) {
					System.err.println("Invalid float is: " + item);
					e.printStackTrace();
					
				}
				
			} else {
				total.length -= 1; //adjusting number to divide sum if we skip a NaN entry (to ensure average ignores NaNs)
			}
			
		}
	}
}
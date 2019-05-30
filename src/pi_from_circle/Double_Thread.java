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
public class Double_Thread implements Runnable{

	final long ITERATIONS;
	int in_circle = 0;
	AtomicLong output;
	
	Double_Thread(long iterations, AtomicLong shared_output)
	{
		ITERATIONS = iterations;
		output = shared_output;
		
	}
	
	
	public void run()
	{
		/* looping */
		for(int i = 0; i <= ITERATIONS; i++)
		{
			in_circle++;
			/*Double x = ThreadLocalRandom.current().nextDouble();
			Double y = ThreadLocalRandom.current().nextDouble();
			
			if((x*x)+(y*y) <= 1)	//checks if value is within the section, based on the formula for an ellipse
				in_circle++;*/
		}
		
		/* adding final result of operation to the shared result */
		output.getAndAdd(in_circle);
	}
	
	
	
}
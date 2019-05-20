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
public class BigDecimal_Thread implements Runnable{

	final long ITERATIONS;
	int in_circle = 0;
	final BigDecimal max = new BigDecimal(1.0);
	final BigDecimal min = new BigDecimal(0.0);
	
	final BigDecimal one = new BigDecimal(1.0); //since this is a quarter circle inside a unit square, thus the formula for an ellipse applies
	AtomicLong output;
	
	BigDecimal_Thread(long iterations, AtomicLong shared_output)
	{
		ITERATIONS = iterations;
		output = shared_output;
		
	}
	
	
	public void run()
	{
		/* looping */
		for(int i = 0; i <= ITERATIONS; i++)
		{
			BigDecimal x = randomBigDecimal(min, max);
			BigDecimal y = randomBigDecimal(min, max);
			
			if((x.multiply(x)).add((y.multiply(y))).compareTo(one) <= 0)
				in_circle++;
		}
		
		/* adding final result of operation to the shared result */
		output.getAndAdd(in_circle);
	}
	
	
	/* make a new decimal within the range of min to max */
	public BigDecimal randomBigDecimal(BigDecimal min, BigDecimal max)
	{
        BigDecimal randFromDouble = new BigDecimal(ThreadLocalRandom.current().nextDouble());
        BigDecimal actualRandomDec = randFromDouble.divide(max,BigDecimal.ROUND_DOWN);
        return min.add(actualRandomDec);
	}
	
}
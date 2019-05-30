package barrier_options;

import java.io.IOException;

/*
 * 
 * Matthew Lee
 * Spring 2019
 * Practicing Monte Carlo Simulation
 * 
 * 
 * Just a class to try running the various implementations from
 * 
 * TODO: calculate how using BigDecimal affects the error of the result (likely statistically insignificant compared
 * 		 to using doubles with a larger number of generated points size)
 * 
 */

public class Initializer {

	public static void main(String[] args) throws InterruptedException, IOException
	{
		int iterations = 100;
		//Double_Approximation.threaded(iterations);
		OpenCL_Accelerated.accelerated(iterations);
	}
}

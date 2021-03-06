package pi_from_circle;

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

public class Simulator {

	public static void main(String[] args) throws InterruptedException, IOException
	{
		long iterations = 10L;
		//Double_Approximation.threaded(iterations);
		OpenCL_mwc64x.accelerated(iterations);
	}
}

package pi_from_circle;

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

	public static void main(String[] args) throws InterruptedException
	{
		Double_Approximation.threaded(10000000000L);
	}
}

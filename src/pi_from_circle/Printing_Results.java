package pi_from_circle;

import bundled_utilities.Time;

public class Printing_Results {

	public static void console(String function, long in_circle, long iterations, long time)
	{
		System.out.println("======================================================");
		double approx = 4.0*((double) in_circle/iterations);
		System.out.println("Function: " + function);
		System.out.println("Points in circle: " + in_circle + " of " + iterations);
		System.out.println("Approximation of pi: " + approx);
		System.out.printf("Error: %f\n", (Math.PI-approx));
		System.out.println("Computation Time: " + Time.from_nano(time));
	}
	
	
	//add printing to file
}

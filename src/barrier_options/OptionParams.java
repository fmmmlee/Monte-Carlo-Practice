package barrier_options;

/*
 * 
 * Matthew Lee
 * Spring 2019
 * 
 * 
 */

public class OptionParams {

	final double mu;
	final double sigma;
	final double years;
	final double start_price;
	final int steps_per_sim;
	final double barrier;
	final double strike_price;
	
	OptionParams(double mu, double sigma, double years, double start_price, int paths_per_year, double barrier, double strike_price)
	{
		this.mu = mu;
		this.sigma = sigma;
		this.years = years;
		this.steps_per_sim = paths_per_year;
		this.barrier = barrier;
		this.start_price = start_price;
		this.strike_price = strike_price;
	}
	
}

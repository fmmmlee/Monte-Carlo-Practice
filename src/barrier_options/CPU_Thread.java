package barrier_options;

import java.util.Random;

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
public class CPU_Thread implements Runnable{


	double[] res;
	double start_price;
	int startingindex;
	int num;
	int per_time;
	double price;
	double sigma;
	double mu;
	double barrier;
	double time;
	
	CPU_Thread(double[] result, int steps_per_sim, double start_price, double sigma, double mu, double time, int threadnum, int jobs, double barrier)
	{
		res = result;
		startingindex = threadnum*jobs;
		num = jobs;
		per_time = steps_per_sim;
		this.start_price = start_price;
		this.sigma = sigma;
		this.mu = mu;
		this.time = time;		
		this.barrier = barrier;
	}
	
	
	public void run()
	{
		double v = mu - (sigma*sigma)/2;
		double dt = (double)time/per_time;
		
		/* looping */
		for(int i = startingindex; i < startingindex+num; i++)
		{
			Random aRand = new Random();
			price = start_price;
			for(int j = 0; j < per_time/2; j++)
			{
				double rand1 = aRand.nextDouble();
				double rand2 = aRand.nextDouble();
				double rand3 = (double) Math.sqrt(-2*Math.log(rand1))*Math.cos(2*Math.PI*rand2);
				double rand4 = (double) Math.sqrt(-2*Math.log(rand1))*Math.sin(2*Math.PI*rand2);
		
				price = price*Math.exp(v*dt + sigma*(Math.sqrt(dt))*rand3);
				if(price < barrier)
					break;
		        price = price*Math.exp(v*dt + sigma*(Math.sqrt(dt))*rand4);
		        if(price < barrier)
					break;
			}
			
			if(per_time % 2 != 0 && price >= barrier)
			{
				double rand1 = aRand.nextDouble();
				double rand2 = aRand.nextDouble();
				double rand3 = (double) Math.sqrt(-2*Math.log(rand1))*Math.cos(2*Math.PI*rand2);
				price = price*Math.exp(v*dt + sigma*(Math.sqrt(dt))*rand3);
			}
			
			if(price >= barrier)
				res[i] = price;
			else
				res[i] = 0.0/0.0;
		}
		
	}
	
	
	
}
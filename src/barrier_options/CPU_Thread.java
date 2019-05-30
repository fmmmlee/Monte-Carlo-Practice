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
 * Using float to see how much faster it is
 * 
 */
public class CPU_Thread implements Runnable{


	float[] res;
	float start_price;
	int startingindex;
	int num;
	int per_time;
	float price;
	float sigma;
	float mu;
	float time;
	
	CPU_Thread(float[] result, int steps_per_sim, float start_price, float sigma, float mu, float time, int threadnum, int jobs)
	{
		res = result;
		startingindex = threadnum*jobs;
		num = jobs;
		per_time = steps_per_sim;
		this.start_price = start_price;
		this.sigma = sigma;
		this.mu = mu;
		this.time = time;		
	}
	
	
	public void run()
	{
		float variance = sigma*sigma;
		float dt = (float)time/per_time;
		
		/* looping */
		for(int i = startingindex; i < startingindex+num; i++)
		{
			Random aRand = new Random();
			price = start_price;
			for(int j = 0; j < per_time/2; j++)
			{
				float rand1 = aRand.nextFloat();
				float rand2 = aRand.nextFloat();
				float rand3 = (float) (variance*(Math.sqrt(-2*Math.log(rand1))*Math.cos(2*Math.PI*rand2)));
				float rand4 = (float) (variance*(Math.sqrt(-2*Math.log(rand1))*Math.sin(2*Math.PI*rand2)));
				price = price + mu*price*dt + sigma*price*rand3;
		        price = price + mu*price*dt + sigma*price*rand4;
			}
			
			res[i] = price;
		}
		
	}
	
	
	
}
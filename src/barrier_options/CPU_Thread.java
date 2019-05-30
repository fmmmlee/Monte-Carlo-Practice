package barrier_options;

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

	double[] arrone;
	double[] arrtwo;
	double[] res;
	int startingindex;
	int num;
	CPU_Thread(double[] arrayone, double[] arraytwo, double[] result, int threadnum, int jobs)
	{
		arrone = arrayone;
		arrtwo = arraytwo;
		res = result;
		startingindex = threadnum*jobs;
		num = jobs;
		
	}
	
	
	public void run()
	{
		/* looping */
		for(int i = startingindex; i < startingindex+num; i++)
		{
			res[i] = Math.sin(arrone[i])*Math.cos(arrtwo[i]);
		}
		
	}
	
	
	
}
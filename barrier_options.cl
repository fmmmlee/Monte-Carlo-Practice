/*
    (c) Matthew Lee
    Spring 2019
    MIT License
*/

#include <mwc64x.cl>

/*RANDOM NOTE TO SELF: at what point is checking for identical dependencies efficient (330)
e.g. if B is dependent on A, and you compute a bunch of Bs from a bunch of As, how to know whether it is worth it to check a new A value against the computed A-B pairs to see if the computation has already been done (might be a lot of pairs, but might also be a huge computation so iterating through list is not significant relatively speaking)*/


kernel void barrier_simulation(global float* result, int steps_per_sim, float start_price, float sigma, float mu, float time, global float* clocks)
{
    //initializing the starting price of the option
    float price = start_price;
    
    //work id
    int gid = get_global_id(0);
    
    //time per iteration
    float dt = (float) time/steps_per_sim;
    
    float variance = sigma*sigma;
    
    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, 0, steps_per_sim);
    
    int start_clock;
    asm("mov.u32 %0, %%clock;" : "=r"(start_clock));

    /* 2 steps per iteration, because the Box-Muller method requires/generates 2 random numbers on the standard normal distribution */
    /*add check for bad divisors or change to use variables out of scope of loop and remove the steps division by 2*/
    for(int i = 0; i < steps_per_sim/2; i++)
    {
        //uniformly distributed 0-1 randoms
        float rand1 = (float)MWC64X_NextUint(&rng)/(float)(UINT_MAX);
        float rand2 = (float)MWC64X_NextUint(&rng)/(float)((UINT_MAX));
        
        //Box–Muller transformation and variance adjustment
        float rand3 = variance*(sqrt(-2*log(rand1))*cos(2*M_PI*rand2));
        float rand4 = variance*(sqrt(-2*log(rand1))*sin(2*M_PI*rand2));
        
        //Euler method
        price = price + mu*price*dt + sigma*price*rand3;
        price = price + mu*price*dt + sigma*price*rand4;
    }
    
    //after completing specified number of steps, input result into array
    result[gid] = price;
    
    int end_clock;
    asm("mov.u32 %0, %%clock;" : "=r"(end_clock));
    
    clocks[gid] = (float)end_clock - (float)start_clock;

}
/*
    (c) Matthew Lee
    Spring 2019
    MIT License
*/

#include <mwc64x.cl>

/*RANDOM NOTE TO SELF: at what point is checking for identical dependencies efficient (330)
e.g. if B is dependent on A, and you compute a bunch of Bs from a bunch of As, how to know whether it is worth it to check a new A value against the computed A-B pairs to see if the computation has already been done (might be a lot of pairs, but might also be a huge computation so iterating through list is not significant relatively speaking)*/


kernel void barrier_simulation(global double* result, global double* payoff, int steps_per_sim, double start_price, double sigma, double mu, double time, double barrier_a, double strike_price)
{
    //initializing the starting price of the option
    double price = start_price;

    double barrier = barrier_a*(exp(+0.5826*sigma*sqrt(time/steps_per_sim)));

    //work id
    int gid = get_global_id(0);

    //time per iteration
    double dt = (double) time/steps_per_sim;

    double v = mu - ((sigma*sigma)/2);

    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, 0, steps_per_sim);

    /* 2 steps per iteration, because the Box-Muller method requires/generates 2 random numbers on the standard normal distribution */
    /*add check for bad divisors or change to use variables out of scope of loop and remove the steps division by 2*/
    for(int i = 0; i < steps_per_sim/2; i++)
    {
        //uniformly distributed 0-1 randoms
        //I think due to the cast to single precision double I'm probably losing a lot of the random period available (since I'm losing a huge chunk of random digits)
        double rand1 = (double)MWC64X_NextUint(&rng)/(UINT_MAX);
        double rand2 = (double)MWC64X_NextUint(&rng)/((UINT_MAX));

        //Box–Muller transformation and variance adjustment
        double rand3 = (sqrt(-2*log(rand1))*cos(2*M_PI*rand2));
        double rand4 = (sqrt(-2*log(rand1))*sin(2*M_PI*rand2));

        //Euler method
        price = price*exp(v*dt + sigma*(sqrt(dt))*rand3);
        if(price < barrier)
            break;
        price = price*exp(v*dt + sigma*(sqrt(dt))*rand4);
        if(price < barrier)
            break;
    }

    //compensating for possible leftover step
    if(steps_per_sim % 2 != 0 && price >= barrier)
    {
       //uniformly distributed 0-1 randoms
        double rand1 = (double)MWC64X_NextUint(&rng)/(double)(UINT_MAX);
        double rand2 = (double)MWC64X_NextUint(&rng)/(double)((UINT_MAX));
        double rand3 = (sqrt(-2*log(rand1))*cos(2*M_PI*rand2));
        price = price*exp(v*dt + sigma*(sqrt(dt))*rand3);
    }

    //after completing specified number of steps, input result into array
    result[gid] = (price >= barrier ? price : (0.0/0.0));
    //input expected payoff into array
    if(price >= barrier)
    {
        double thispayoff = price - strike_price;
        payoff[gid] = (thispayoff > 0 ? thispayoff*exp((-mu)*time) : 0.0);
    } else {
        payoff[gid] = 0.0;
    }

}




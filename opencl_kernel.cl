/*
    Matthew Lee
    Spring 2019

*/

#include <mwc64x.cl>

kernel void Trig(global const float *one, global const float *two, global const float *result, global const int size)
{
    mwc64x_state_t rng;
    ulong samplesPerStream=size/get_global_size(0);
    MWC64X_SeedStreams(&rng, baseOffset, 2*samplesPerStream);
    int gid = get_global_id(0);
    ulong x=MWC64X_NextUint(&rng);
    ulong y=MWC64X_NextUint(&rng);
    ulong x2=x*x;
    ulong y2=y*y;
    
    for(int i = gid; i < gid+samplesPerStream; i++){
        if(x2+y2 >= x2)
            result[gid] = 1;
        else
            result[gid] = 0; 
    }
}
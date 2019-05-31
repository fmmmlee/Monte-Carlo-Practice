# Monte-Carlo-Practice
Just getting in some practice trying out different methods of calculating various things using Monte Carlo simulations.

## TODOS

### barrier_options
- implement barriers into option price calculation in kernel
- implement Kahan summation in kernel
- add option to read/write metrics to log file
- possibly change Box-Muller transformation in kernel to something that doesn't require using 2 random numbers (if such an algorithm exists and isn't a performance loss - I'm not hopeful)

### bundled_utilities
- implement Kahan summation in array average function (and the threads it spawns)

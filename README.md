# Monte-Carlo-Practice
Just getting in some practice on different concepts (Monte Carlo Simulations, OpenCL, multithreading, lots of stuff)
### Example output:
(there's some discrepancy between the CPU and GPU functions I run, which should explain the fairly widely disparate estimates - I need to debug that)
<p align="left">
  <img src="https://user-images.githubusercontent.com/30479162/58740807-969b4700-83c7-11e9-866b-ae35fc375e60.JPG" width="1000" title="a sample run">
</p>

## TODOS

### barrier_options
- implement barriers into option price calculation in kernel
- implement Kahan summation in kernel
- add option to read/write metrics to log file
- identify difference between CPU and GPU versions that causes CPU to have consistently lower estimate
- possibly change Box-Muller transformation in kernel to something that doesn't require using 2 random numbers (if such an algorithm exists and isn't a performance loss - I'm not hopeful)

### bundled_utilities
- implement Kahan summation in array average function (and the threads it spawns)

package barrier_options;


import static org.jocl.CL.*;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;

import org.apache.commons.io.FileUtils;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import bundled_utilities.Average_of_Array;
import bundled_utilities.Time;

//TODO: Change kernel to use Kahan summation
//TODO: Add option to write to file/database for metrics, with codes like [CLOCKS] and [TOTAL] for easier parsing later
//TODO: Graphing utility
//TODO: use GMP for GPU accelerated array sum
//TODO: Calculate total error from ideal based on floating point error in single and double precision floats, number of iterations, number of path divergences per iteration, etc.
//full explanation of estimating down-and-out exotic option prices can be found at https://pdfs.semanticscholar.org/542f/6e1e9338632e3bc5b56dad2515854e34190f.pdf


/*
 * Things with a random // after them are things that I don't fully understand and need to research/look @ their docs
 */
public class OpenCL_Accelerated {

	public static void accelerated(int count_in, OptionParams requested) throws IOException, InterruptedException
	{
		/* variables used in kernel args and run configuration */
		final double mu = requested.mu;
		final double sigma = requested.sigma;
		final double years = requested.years;
		final double start_price = requested.start_price;
		final int num_simulations = count_in;
		final int steps_per_sim = requested.steps_per_sim;
		final double strike_price = requested.strike_price;
		final double barrier = requested.barrier;
		double result[] = new double[num_simulations];
		double payoff[]= new double[num_simulations];
		
		/* opencl setup */
		final int platformIndex = 0;
		final int dimensions = 1;
		final long deviceType = CL_DEVICE_TYPE_ALL;
		final int deviceIndex = 0;
		final String libraries = "-I C:\\OpenCL_Libraries\\mwc64x-v0\\mwc64x\\cl";

		File kernel_file = new File("barrier_options.cl");
		
		String kernel = FileUtils.readFileToString(kernel_file, StandardCharsets.UTF_8);
		
		/* padding */
		System.out.println("======================================================");
		System.out.println("======================================================");
		
		/**** error messages, metrics, etc ****/
		setExceptionsEnabled(true);
		cl_event timing[] = new cl_event[]{new cl_event()};
		int err[] = new int[1];
		/**** end this metrics section ****/
		
		
		/* argument pointers */
		Pointer resultPtr = Pointer.to(result);
		Pointer payPtr = Pointer.to(payoff);
		Pointer strikePtr = Pointer.to(new double[]{strike_price});
		Pointer perPtr = Pointer.to(new int[]{steps_per_sim});
		Pointer startPtr = Pointer.to(new double[]{start_price});
		Pointer sigmaPtr = Pointer.to(new double[]{sigma});
		Pointer muPtr = Pointer.to(new double[]{mu});
		Pointer yearsPtr = Pointer.to(new double[]{years});
		Pointer barPtr = Pointer.to(new double[]{barrier});
		
		/* number of platforms */
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];
		
		/* platform id */
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];		
		
		/* properties for the context */
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
		
		/* number of devices */
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		
		/* getting device ID(s) */
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];
		
		/* creating context */
		cl_context context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
		
		/* creating queue to put commands into */
		//to not watch metrics/queue profile, remove next two lines and set last 2 parameters of the command queue constructor to null
		cl_queue_properties for_timing = new cl_queue_properties();
		for_timing.addProperty(CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE);
		cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, device, for_timing, err);
		
		/* memory objects for reading data from GPU after execution */
		cl_mem memObjects[] = new cl_mem[2];
		memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_simulations*Sizeof.cl_double, null, null);
		memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_simulations*Sizeof.cl_double, null, null);
		
		/* creating program */
		cl_program program = clCreateProgramWithSource(context, 1, new String[] {kernel}, null, null);
		clBuildProgram(program, 0, null, libraries, null, null);
			
		/* setting up kernel and adding arguments */
		cl_kernel clKernel = clCreateKernel(program, "barrier_simulation", null);
		
		long multiple[] = new long[1];
		Pointer multPtr = Pointer.to(multiple);
		clGetKernelWorkGroupInfo(clKernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, Sizeof.cl_long, multPtr, null);
		long preferred_mult = multiple[0];
		System.out.println("[INFO] Preferred work group size multiple: " + preferred_mult);
		
		long max[] = new long[1];
		Pointer maxPtr = Pointer.to(max);
		clGetKernelWorkGroupInfo(clKernel, device, CL_KERNEL_WORK_GROUP_SIZE, Sizeof.cl_long, maxPtr, null);
		long max_val = max[0];
		System.out.println("[INFO] Maximum work group size: " + max_val);
		
		//TODO: set local work size to greatest common factor of maximum and preferred mult
		clSetKernelArg(clKernel, 0, Sizeof.cl_mem, Pointer.to(memObjects[0]));
		clSetKernelArg(clKernel, 1, Sizeof.cl_mem, Pointer.to(memObjects[1]));
		clSetKernelArg(clKernel, 2, Sizeof.cl_uint, perPtr);
		clSetKernelArg(clKernel, 3, Sizeof.cl_double, startPtr);
		clSetKernelArg(clKernel, 4, Sizeof.cl_double, sigmaPtr);
		clSetKernelArg(clKernel, 5, Sizeof.cl_double, muPtr);
		clSetKernelArg(clKernel, 6, Sizeof.cl_double, yearsPtr);
		clSetKernelArg(clKernel, 7, Sizeof.cl_double, barPtr);
		clSetKernelArg(clKernel, 8, Sizeof.cl_double, strikePtr);
		
		/* global/local work sizes */
		long global_work_num_simulations[] = new long[]{(long) (num_simulations/(Math.pow(10, 1.0-dimensions)))};
		long local_work_num_simulations[] = new long[]{512}; //TODO: Compare runyearss with different multiples of 1024 as the local work sizes and different amounts of data to work with
		
		/* getting the name of the platform */
		long size_of_platform_name[] = new long[1];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, null, size_of_platform_name);
		byte platform_name_buf[] = new byte[(int)size_of_platform_name[0]];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_buf.length, Pointer.to(platform_name_buf), null);
		String platform_name = new String(platform_name_buf, 0, platform_name_buf.length-1);
		
		/* getting the name of the device */
		long size_of_device_name[] = new long[1];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, size_of_device_name);
		byte device_name_buf[] = new byte[(int)size_of_device_name[0]];
		clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_buf.length, Pointer.to(device_name_buf), null);
		String device_name = new String(device_name_buf, 0, device_name_buf.length-1);
		
		//observation/metrics
		System.out.println("[STATUS] Executing OpenCL Kernel on a [" + device_name + "] using the [" + platform_name + "] platform");
		long start_time = System.nanoTime();
		
		/* putting kernel in command queue for execution (no other items in queue, so automatically executes immediately) */
		clEnqueueNDRangeKernel(commandQueue, clKernel, dimensions, null, global_work_num_simulations, local_work_num_simulations, 0, null, timing[0]);
		
		/* waiting for kernel to execute */
		clWaitForEvents(1, timing);
		clFinish(commandQueue);
		
		/* post-execution (this call blocks because I don't always leave a wait call before it), reading memory from buffer into pointer */
		clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE, 0, (num_simulations*Sizeof.cl_double), resultPtr, 0, null, null);
		clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, (num_simulations*Sizeof.cl_double), payPtr, 0, null, null);
		
		/* output/cleanup */
		long end_time = System.nanoTime();
		long total_time = end_time - start_time;
		
		/* releasing OpenCL objects */
		clReleaseMemObject(memObjects[0]);
		clReleaseKernel(clKernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
		
		
		
		/* observation */
		System.out.println("[STATUS] Kernel and Cleanup Finished");
		
		
		/**** metrics again ****/
		long kernel_start[] = new long[]{0};
		long kernel_end[] = new long[]{0};
		clGetEventProfilingInfo(timing[0], CL_PROFILING_COMMAND_START, (long)Sizeof.cl_long, Pointer.to(kernel_start), null);
		clGetEventProfilingInfo(timing[0], CL_PROFILING_COMMAND_END, (long)Sizeof.cl_long, Pointer.to(kernel_end), null);
		long kernel_time = kernel_end[0]-kernel_start[0];
		/**** end metrics section ****/
		
		
		/********printing results********/
		
		System.out.println("======================================================");
		System.out.println("[GPU] Length of each simulation: " + years + " years");
		System.out.println("[GPU] Price path decisions calculated at " + steps_per_sim + " intervals.");
		System.out.println("======================================================");
		System.out.println("[GPU] Number of individual simulations run: " + num_simulations);
		System.out.println("[GPU] Average time per calculation:" + Time.from_nano(kernel_time/num_simulations));
		System.out.println("[GPU] Kernel Execution Time: " + Time.from_nano(kernel_time));
		System.out.println("[GPU] Time from before queueing command in Java to after reading GPU memory in Java:" + Time.from_nano(total_time));
		System.out.println("[GPU] Application and GPU command queue overhead based on the above two times is:" + Time.from_nano(total_time - kernel_time));
		System.out.println("======================================================");
		
		/* prices - concurrent average function */
		//TODO: just do this averaging on the GPU too
		long start_mult = System.nanoTime();
		BigDecimal res = Average_of_Array.avg_double(result);
		long end_mult = System.nanoTime();
		
		BigDecimal pay = Average_of_Array.avg_double(payoff);
		
		System.out.println("[GPU] Average projected option price after the time period specified: " + res + " (average calculated using concurrent function)");
		System.out.println("[GPU] Time to compute average:" + Time.from_nano(end_mult - start_mult));
		System.out.println("[GPU] Projected option payoff after the time period specified: " + pay + " (average calculated using concurrent function)");
		System.out.println("======================================================");
		System.out.println("======================================================");
	}
}

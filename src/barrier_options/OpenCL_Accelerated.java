package barrier_options;


import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.apache.commons.io.FileUtils;
import org.jocl.CL;
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

//TODO: Implement barriers into the kernel - it's in the package name, after all

/*
 * Things with a random // after them are things that I don't fully understand and need to research/look @ their docs
 */
public class OpenCL_Accelerated {

	public static void accelerated(int count_in) throws IOException
	{
		final int platformIndex = 0;				//
		final long deviceType = org.jocl.CL.CL_DEVICE_TYPE_ALL; //
		final int deviceIndex = 0;
		final String libraries = "-I C:\\OpenCL_Libraries\\mwc64x-v0\\mwc64x\\cl";

		File kernel_file = new File("barrier_options.cl");
		
		String kernel = FileUtils.readFileToString(kernel_file, StandardCharsets.UTF_8);
		
		/* program arguments and run configuration */
		final float mu = 0.1f;
		final float sigma = 0.1f;
		final float time = 1.0f;
		final float start_price = 100.0f;
		final int num_simulations = count_in;
		final int steps_per_sim = 365;
		float result[] = new float[num_simulations];
		
		
		/**** error messages, metrics, etc ****/
		CL.setExceptionsEnabled(true);
		cl_event timing[] = new cl_event[]{new cl_event()};
		int err[] = new int[1];
		float clocks[] = new float[num_simulations];
		Pointer clockPtr = Pointer.to(clocks);
		/**** end this metrics section ****/
		
		
		/* argument pointers */
		Pointer resultPtr = Pointer.to(result);
		Pointer perPtr = Pointer.to(new int[]{steps_per_sim});
		Pointer startPtr = Pointer.to(new float[]{start_price});
		Pointer sigmaPtr = Pointer.to(new float[]{sigma});
		Pointer muPtr = Pointer.to(new float[]{mu});
		Pointer timePtr = Pointer.to(new float[]{time});
		
		/* number of platforms */
		int numPlatformsArray[] = new int[1];
		CL.clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];
		
		/* platform id */
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		CL.clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];
		
		/* properties for the context */
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
		
		/* number of devices */
		int numDevicesArray[] = new int[1];
		CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		
		/* getting device ID(s) */
		cl_device_id devices[] = new cl_device_id[numDevices];
		CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];
		
		/* creating context */
		cl_context context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
		
		/* creating queue to put commands into */
		//to not watch metrics/queue profile, remove next two lines and set last 2 parameters of the command queue constructor to null
		cl_queue_properties for_timing = new cl_queue_properties();
		for_timing.addProperty(CL.CL_QUEUE_PROPERTIES, CL.CL_QUEUE_PROFILING_ENABLE);
		cl_command_queue commandQueue = CL.clCreateCommandQueueWithProperties(context, device, for_timing, err);
		
		/* memory objects for reading data from GPU after execution */
		cl_mem memObjects[] = new cl_mem[2];
		memObjects[0] = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, num_simulations*Sizeof.cl_float, null, null);
		memObjects[1] = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, num_simulations*Sizeof.cl_float, null, null); //metrics - array with number of clock cycles elapsed for each kernel execution
		
		/* creating program */
		cl_program program = CL.clCreateProgramWithSource(context, 1, new String[] {kernel}, null, null);
		CL.clBuildProgram(program, 0, null, libraries, null, null);
			
		/* setting up kernel and adding arguments */
		cl_kernel clKernel = CL.clCreateKernel(program, "barrier_simulation", null);
		CL.clSetKernelArg(clKernel, 0, Sizeof.cl_mem, Pointer.to(memObjects[0]));
		CL.clSetKernelArg(clKernel, 1, Sizeof.cl_uint, perPtr);
		CL.clSetKernelArg(clKernel, 2, Sizeof.cl_float, startPtr);
		CL.clSetKernelArg(clKernel, 3, Sizeof.cl_float, sigmaPtr);
		CL.clSetKernelArg(clKernel, 4, Sizeof.cl_float, muPtr);
		CL.clSetKernelArg(clKernel, 5, Sizeof.cl_float, timePtr);
		CL.clSetKernelArg(clKernel, 6, Sizeof.cl_mem, Pointer.to(memObjects[1]));
		
		/* global/local work sizes */
		long global_work_num_simulations[] = new long[]{num_simulations};
		long local_work_num_simulations[] = new long[]{1}; //may change this
		
		//for observation/metrics
		System.out.println("[STATUS] Executing OpenCL Kernel");
		long start_time = System.nanoTime();
		
		/* putting kernel in command queue for execution (no other items in queue, so automatically executes immediately) */
		CL.clEnqueueNDRangeKernel(commandQueue, clKernel, 1, null, global_work_num_simulations, local_work_num_simulations, 0, null, timing[0]);
		
		/* waiting for kernel to execute */
		CL.clWaitForEvents(1, timing);
		CL.clFinish(commandQueue);
		
		/* post-execution (this call blocks because I don't always leave a wait call before it), reading memory from buffer into pointer */
		CL.clEnqueueReadBuffer(commandQueue, memObjects[0], CL.CL_TRUE, 0, (num_simulations*Sizeof.cl_float), resultPtr, 0, null, null);
		CL.clEnqueueReadBuffer(commandQueue, memObjects[1], CL.CL_TRUE, 0, (num_simulations*Sizeof.cl_float), clockPtr, 0, null, null);
		
		/* output/cleanup */
		long end_time = System.nanoTime();
		long total_time = end_time - start_time;
		
		/* releasing OpenCL objects */
		CL.clReleaseMemObject(memObjects[0]);
		CL.clReleaseMemObject(memObjects[1]);
		CL.clReleaseKernel(clKernel);
		CL.clReleaseProgram(program);
		CL.clReleaseCommandQueue(commandQueue);
		CL.clReleaseContext(context);
		
		
		
		/* observation */
		System.out.println("[STATUS] Kernel and Cleanup Finished");
		
		
		/**** metrics again ****/
		long kernel_start[] = new long[]{0};
		long kernel_end[] = new long[]{0};
		CL.clGetEventProfilingInfo(timing[0], CL.CL_PROFILING_COMMAND_START, (long)Sizeof.cl_long, Pointer.to(kernel_start), null);
		CL.clGetEventProfilingInfo(timing[0], CL.CL_PROFILING_COMMAND_END, (long)Sizeof.cl_long, Pointer.to(kernel_end), null);
		long kernel_time = kernel_end[0]-kernel_start[0];
		/**** end metrics section ****/
		
		
		/********printing results********/
		/* prices */
		long avg_price = 0;
		for(float price : result)
		{
			avg_price += price;
		}
		
		/* clocks */
		long avg_clocks = 0;
		for(float clock : clocks)
		{
			avg_clocks += clock;
		}
		
		System.out.println("======================================================");
		System.out.println("Length of each simulation: " + time + " years, with randomness inserted at " + steps_per_sim + " intervals.");
		System.out.println("======================================================");
		System.out.println("Number of individual simulations run: " + num_simulations);
		System.out.println("Average clocks per work item: " + (float)avg_clocks/num_simulations + " (the GTX 1060 I test this on should perform about 1.404 billion clocks per second)");
		System.out.println("Kernel Execution Time: " + bundled_utilities.Time.from_nano(kernel_time));
		System.out.println("Time from before queueing command in Java to after reading GPU memory in Java: " + bundled_utilities.Time.from_nano(total_time));
		System.out.println("Overhead based on the above two times is: " + bundled_utilities.Time.from_nano(total_time - kernel_time));
		System.out.println("======================================================");
		System.out.println("Projected option price after 1 year (split into 365 days): " + (float)avg_price/num_simulations);
	}
}

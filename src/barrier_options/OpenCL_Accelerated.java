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
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;


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
		
		/* to see error messages */
		CL.setExceptionsEnabled(true);
		
		/* program arguments and run configuration */
		final float mu = 0.1f;
		final float sigma = 0.1f;
		final float time = 1.0f;
		final float start_price = 100.0f;
		final int num_simulations = count_in;
		final int steps_per_sim = 365;
		float result[] = new float[num_simulations];
		
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
		
		//
		cl_context context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
		
		//
		cl_command_queue commandQueue = CL.clCreateCommandQueueWithProperties(context, device, null, null);
		
		/* memory objects for program arguments */
		cl_mem memObjects[] = new cl_mem[1];
		memObjects[0] = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, num_simulations*Sizeof.cl_float, null, null);
		
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
		
		/* global/local work sizes */
		long global_work_num_simulations[] = new long[]{num_simulations};
		long local_work_num_simulations[] = new long[]{1}; //may change this
		
		//for observation/metrics
		System.out.println("Executing OpenCL Kernel");
		long start_time = System.nanoTime();
		
		/* putting kernel in command queue for execution (no other items in queue, so automatically executes immediately) */
		CL.clEnqueueNDRangeKernel(commandQueue, clKernel, 1, null, global_work_num_simulations, local_work_num_simulations, 0, null, null);	
		/* post-execution (this call blocks), reading memory from buffer into pointer */
		CL.clEnqueueReadBuffer(commandQueue, memObjects[0], CL.CL_TRUE, 0, (num_simulations*Sizeof.cl_float), resultPtr, 0, null, null);
		
		
		/* releasing OpenCL objects */
		CL.clReleaseMemObject(memObjects[0]);
		CL.clReleaseKernel(clKernel);
		CL.clReleaseProgram(program);
		CL.clReleaseCommandQueue(commandQueue);
		CL.clReleaseContext(context);
		
		/* output/cleanup */
		long end_time = System.nanoTime();
		for(float single : result)
		{
			System.out.println(single);
		}
		
		System.out.println(bundled_utilities.Time.from_nano(end_time-start_time));
		
	}	
	
}

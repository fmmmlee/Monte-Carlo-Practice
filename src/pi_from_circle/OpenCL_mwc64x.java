package pi_from_circle;


import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.apache.commons.io.FileUtils;
import org.jocl.*;
import org.jocl.cl_context;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;


/*
 * Things with a random // after them are things that I don't fully understand and need to research/look @ their docs
 */
public class OpenCL_mwc64x {

	//function declaration in kernel: kernel void EstimatePi(uint n, ulong baseOffset, __global uint *acc)
	
	
	public static void accelerated(long count_in) throws IOException
	{
		final long iterations[] = {count_in}; //number of random points to generate
		final int platformIndex = 0;				//
		final long deviceType = org.jocl.CL.CL_DEVICE_TYPE_ALL; //
		final int deviceIndex = 0;
		final String libraries = "-I C:\\OpenCL_Libraries\\mwc64x-v0\\mwc64x\\cl";
		
		
		
		
		/*
		 * 
		 * Kernel source code (all credit to MWC64x, see file)
		 * 
		 */
		File kernel_file = new File("opencl_kernel.cl"); //I might just put the whole program in a string at the top and see if that gives
		//a tiny performance improvement, since it's removing the file read overhead
		
		String kernel = FileUtils.readFileToString(kernel_file, StandardCharsets.UTF_8);
		
		long approx[] = new long[1]; //empty array which will hold approximation of pi
		
		//Pointers that the kernel will access
		Pointer apxptr = Pointer.to(approx);
		
		CL.setExceptionsEnabled(true);
		
		int numPlatformsArray[] = new int[1];
		CL.clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];
		
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		CL.clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];
		
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
		
		int numDevicesArray[] = new int[1];
		CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		
		cl_device_id devices[] = new cl_device_id[numDevices];
		CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];
		
		cl_context context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
		
		cl_command_queue commandQueue = CL.clCreateCommandQueueWithProperties(context, device, null, null);
		
		cl_mem memObjects[] = new cl_mem[2];
		
		memObjects[0] = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_long, apxptr, null);
		
		cl_program program = CL.clCreateProgramWithSource(context, 1, new String[] {kernel}, null, null);
		
		CL.clBuildProgram(program, 0, null, libraries, null, null);
		
		cl_kernel clKernel = CL.clCreateKernel(program, "EstimatePi", null);
		
		CL.clSetKernelArg(clKernel, 0, Sizeof.cl_mem, Pointer.to(new long[]{count_in}));
		CL.clSetKernelArg(clKernel, 1, Sizeof.cl_mem, Pointer.to(new long[]{0}));
		CL.clSetKernelArg(clKernel, 2, Sizeof.cl_mem, Pointer.to(memObjects[0]));
		
		long global_work_size[] = new long[]{count_in};
		long local_work_size[] = new long[]{10};
		
		System.out.println("Executing OpenCL Kernel");
		long start_time = System.nanoTime();
		CL.clEnqueueNDRangeKernel(commandQueue, clKernel, 1, null, global_work_size, local_work_size, 0, null, null);
		
		CL.clEnqueueReadBuffer(commandQueue, memObjects[0], CL.CL_TRUE, 0, Sizeof.cl_long, apxptr, 0, null, null);
		
		CL.clReleaseMemObject(memObjects[0]);
		CL.clReleaseKernel(clKernel);
		CL.clReleaseProgram(program);
		CL.clReleaseCommandQueue(commandQueue);
		CL.clReleaseContext(context);
		
		long end_time = System.nanoTime();
		Printing_Results.console("GPU Accelerated OpenCL Kernel", approx[0], iterations[0], end_time-start_time);

		
		
	}	
	
}

package pi_from_circle;

import org.jocl.*;
public class SystemStats {


	public static void main(String[] args)
	{
		System.out.println("Maximum work group size: " + CL.CL_DEVICE_MAX_WORK_GROUP_SIZE);
		System.out.println("Maximum work item dimensions: " + CL.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
		System.out.println("Maximum work item sizes: " + CL.CL_DEVICE_MAX_WORK_ITEM_SIZES);
		System.out.println("CL kernel work group size: " + CL.CL_KERNEL_WORK_GROUP_SIZE);
	}

}

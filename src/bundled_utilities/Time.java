package bundled_utilities;

/*
 * Copied from my PrettyUnits package - when it's properly fleshed out I'll give it its own repository
 * 
 * Matthew Lee
 * Spring 2019
 * 
 * 
 */

public class Time {

	public static String from_nano(long input)
	{
		long seconds = input/1000000000;
		input -= seconds*1000000000;
		
		long minutes = seconds/60;
		seconds -= minutes*60;
		
		long hours = minutes/60;
		minutes -= hours*60;
		
		long milliseconds = input/1000000;
		input -= milliseconds*1000000;
		
		long microseconds = input/1000;
		input -= microseconds*1000;
		
		String returned = "";
		if(hours != 0)
			returned += (hours + " hrs");
		if(minutes != 0)
			returned += (" " + minutes + " min");
		if(seconds != 0)
			returned += (" " + seconds + " sec");
		if(milliseconds != 0)
			returned+= (" " + milliseconds + " millisec");
		if(microseconds != 0)
			returned += (" " + microseconds + " microsec");
		if(input != 0)
			returned += (" " + input + " ns");		
		
		return returned;
	}
}

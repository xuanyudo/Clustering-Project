import java.util.*;
import java.io.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;

@SuppressWarnings("deprecation")
public class KM_try {
	
	public static void main(String[] args) throws Exception {


		Integer iter_num = 0;
		Boolean completed = false, comp_sign = false;
		HashMap<Integer, ArrayList<Double>> prevcents = null;
		HashMap<Integer, ArrayList<Double>> new_cen = null;

		String in_path = args[0];
		String out_path = args[1];
		String init_cent_path = args[2];
		String newLine = System.getProperty("line.separator");
		while(true) {
			
			System.out.println("Iteration" + iter_num);
			
			Configuration conf= new Configuration();
			
			
			if(iter_num == 0) {
				conf.set("centFilePath", init_cent_path);
			}else {
				conf.set("centFilePath", out_path + (iter_num - 1) + output_file);
			}
			
			
			
			if(comp_sign) {
				conf.set("isFinal", "true");
			}else {
				conf.set("isFinal", "false");
			}
			
			
			Job job = new Job(conf,"KMeans");
			job.setJarByClass(KM_try.class);
			job.setMapperClass(Map.class);
			job.setReducerClass(Reduce.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			FileInputFormat.addInputPath(job, new Path(in_path));
			
			
			if(comp_sign) {
				FileOutputFormat.setOutputPath(job, new Path(out_path + "_final"));
			}else {
				FileOutputFormat.setOutputPath(job, new Path(out_path + iter_num));
			}
			completed = job.waitForCompletion(true);
			
			
			
			if (completed) {
				if(comp_sign) {
					break;
				}
				
				new_cen = new HashMap<Integer, ArrayList<Double>>();
				prevcents = new HashMap<Integer, ArrayList<Double>>();
				
				
				
				FileSystem fileSys = FileSystem.get(new Configuration());
				ArrayList<Double> cent = new ArrayList<Double>();
				BufferedReader buffRdr = null;
				String line = "";
				Path filePath = null;
				
				
				// Read the previous cents from reducer output.
				if(iter_num == 0) {
					filePath = new Path(init_cent_path);
				}else {
					filePath = new Path(out_path + (iter_num - 1) + output_file);
				}
				
				
				buffRdr = new BufferedReader(new InputStreamReader(fileSys.open(filePath)));
				line = buffRdr.readLine();
				while (line != null) {
					cent = new ArrayList<Double>();
					String[] strArr = line.split("\t| ");
					for(int i = 1; i < strArr.length; i++) {
						cent.add(Double.parseDouble(strArr[i]));
					}
					prevcents.put(Integer.parseInt(strArr[0]), cent);
					line = buffRdr.readLine();
				}
				buffRdr.close();
				
				
				// new computed cents based on reducer output.
				filePath = new Path(out_path + iter_num + output_file);
				buffRdr = new BufferedReader(new InputStreamReader(fileSys.open(filePath)));
				line = buffRdr.readLine();
				
				
				while (line != null) {
					cent = new ArrayList<Double>();
					String[] strArr = line.split("\t| ");
					for(int i = 1; i < strArr.length; i++) {
						cent.add(Double.parseDouble(strArr[i]));
					}
					new_cen.put(Integer.parseInt(strArr[0]), cent);
					line = buffRdr.readLine();
				}
				buffRdr.close();
				
				
				if(cp_cent(prevcents, new_cen)) {
					comp_sign = true;
				}
				iter_num++;
	       }
		}
		System.exit(comp_sign ? 0 : 1);
	}
	
	

	public static class Map extends Mapper<LongWritable,Text,IntWritable,Text> {
		
		private static HashMap<Integer, ArrayList<Double>> cents = new HashMap<Integer, ArrayList<Double>>();	

		protected void setup(Context context) throws IOException, InterruptedException {
			
	       Configuration conf = context.getConfiguration();
	       String centFilePath = conf.get("centFilePath");
	       FileSystem fileSys = FileSystem.get(conf);
	       BufferedReader buffRdr = new BufferedReader(new InputStreamReader(fileSys.open(new Path(centFilePath))));
			String line = buffRdr.readLine();
			
			
			ArrayList<Double> cent = new ArrayList<Double>();
			while (line != null) {
				cent = new ArrayList<Double>();
				String[] strArr = line.split("\t| ");
				for(int i = 1; i < strArr.length; i++) {
					cent.add(Double.parseDouble(strArr[i]));
				}
				cents.put(Integer.parseInt(strArr[0]), cent);
				line = buffRdr.readLine();
			}
			buffRdr.close();
	   }
		
		public void map(LongWritable key, Text value,Context context) throws IOException,InterruptedException{
			
			String[] fetr = value.toString().split(sepera);
			ArrayList<Double> gene = new ArrayList<Double>();
			for(int i = 2; i < fetr.length; i++) {
				gene.add(Double.parseDouble(fetr[i]));
			}
			Integer centId = closestcent(gene);
			context.write(new IntWritable(centId), value);
		}
		
		public static Integer closestcent(ArrayList<Double> gene) {
			
			Integer mincentId = null;
			Double min_len = null, eu_len = null;
			for(Integer cen_ind : cents.keySet()) {
				eu_len = 0.0;
				for(int i = 0; i < gene.size(); i++) {
					eu_len += Math.pow(gene.get(i)-cents.get(cen_ind).get(i), 2);
				}
				eu_len = Math.sqrt(eu_len);
				if(min_len == null || eu_len < min_len) {
					min_len = eu_len;
					mincentId = cen_ind;
				}
			}	
			return mincentId;
		}
		
       public static Integer closestcent_L1_norm(ArrayList<Double> gene) {
			
			Integer mincentId = null;
			Double min_len = null, eu_len = null;
			for(Integer cen_ind : cents.keySet()) {
				eu_len = 0.0;
				for(int i = 0; i < gene.size(); i++) {
					eu_len += Math.abs(gene.get(i)-cents.get(cen_ind).get(i));
				}
				if(min_len == null || eu_len < min_len) {
					min_len = eu_len;
					mincentId = cen_ind;
				}
			}	
			return mincentId;
		}
		
       
       public static Integer closestcent_Linf_norm(ArrayList<Double> gene) {
			
			Integer mincentId = null;
			Double min_len = null, eu_len = null;
			for(Integer cen_ind : cents.keySet()) {
				eu_len = 0.0;
				for(int i = 0; i < gene.size(); i++) {
					eu_len = Math.max(Math.abs(gene.get(i)-cents.get(cen_ind).get(i)), eu_len);
				}
				if(min_len == null || eu_len < min_len) {
					min_len = eu_len;
					mincentId = cen_ind;
				}
			}	
			return mincentId;
		}
		
		
		
		
	}
	
	public static class Reduce extends Reducer<IntWritable,Text,IntWritable,Text> {
		
		public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException,InterruptedException {
		
			String[] fetr = null;
			String out = "";
			Configuration conf = context.getConfiguration();
			
			ArrayList<ArrayList<Double>> cluster = new ArrayList<ArrayList<Double>>();
			ArrayList<Double> gene = new ArrayList<Double>();

			String isFinal = conf.get("isFinal");
	        System.out.println(isFinal);
	        
	        if(isFinal.equals("false")) {
				for(Text str_val: values){
					fetr = str_val.toString().split("\t| ");
					gene = new ArrayList<Double>();
					for(int i = 2; i < fetr.length; i++) {
						gene.add(Double.parseDouble(fetr[i]));
					}
					cluster.add(gene);
				}
				
				ArrayList<Double> cents = cal_cents(cluster);
				if(cents.size() > 0) {
					for(Double attr : cents) {
						out += String.valueOf(attr) + " ";
					}
					context.write(key, new Text(out.substring(0, out.length() - 1)));
				}
	        }else if(isFinal.equals("true")) {
		       	for(Text str_val: values){
		       		
		       		fetr = str_val.toString().split("\t| ");
		       		out = fetr[0] + " ";
		       		
		       		for(int i = 2; i < fetr.length; i++) {
		       			out += fetr[i] + " ";
						}
		       		context.write(key, new Text(out.substring(0, out.length() - 1)));
				}
	        }
		}
		
		
		public static ArrayList<Double> cal_cents(ArrayList<ArrayList<Double>> cluster){
		

			Integer val_len = cluster.get(0).size();
			ArrayList<Double> cents = new ArrayList<Double>(Arrays.asList(new Double[val_len]));
			Double val_sum = null;
			for(int i = 0; i < val_len; i++) {
				val_sum = 0.0;
				for(ArrayList<Double> gene : cluster) {
					val_sum += gene.get(i);
				}
				cents.set(i, val_sum/cluster.size());
			}
			return cents;
		}
	}

	
	public static Boolean cp_cent(HashMap<Integer, ArrayList<Double>> old_cen, HashMap<Integer, ArrayList<Double>> new_cen) {    
		
		for(Integer cen_ind : old_cen.keySet()) {
			if(!old_cen.get(cen_ind).equals(new_cen.get(cen_ind))) {
				return false;
			}
		}
		return true;
	}
	
	public static final String sepera = "\t| ";
	public static final String output_file = "/part-r-00000";
	
	
}




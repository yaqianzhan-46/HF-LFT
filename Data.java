package LLFT;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class Data {

	public static String trainFilePath="";
	public static String testFilePath="";
	
	
	public static HashMap<Integer, ArrayList<TensorTuple>> trainData=new HashMap<>();
	//public static ArrayList<TensorTuple> trainData1=new ArrayList<>();
	public static ArrayList<TensorTuple> testData=new ArrayList<>();//
	public static int trainCount=0, testCount=0;
	public static int[] iTrainCount;
	
	public static int maxUid=Integer.MIN_VALUE, maxSid=Integer.MIN_VALUE, maxTid=Integer.MIN_VALUE;
	public static int minUid=Integer.MAX_VALUE, minSid=Integer.MAX_VALUE, minTid=Integer.MAX_VALUE;
	public static double maxValue=Double.MIN_VALUE, minValue=Double.MAX_VALUE;

	public static double[][] U,S,T;	
	

	public static int rank=20;
	public static double lambda=0.01;
	public static int trainRound=1000;
	public static double p=1;
	
	public static HashSet<Integer> u_set;
	
	public static HashMap<Integer, ArrayList<GradientTuple>> Gradient=new HashMap<>();
	

	public static double[][] SGradient, TGradient;
	
	// for centralized MANNER
	public static double[][] delta_U,delta_S,delta_T;
	
	//test
	public static double lastTimeRMSE=Double.MAX_VALUE;
	public static double thisTimeRMSE=Double.MAX_VALUE-10;
	public static int threshold=5;
	public static double minRMSE=Double.MAX_VALUE;
	
	public static double lastTimeMAE=Double.MAX_VALUE;
	public static double thisTimeMAE=Double.MAX_VALUE-10;
	public static double minMAE=Double.MAX_VALUE;
	public static int count=0;
	public static int count1=1;
	public static String init;

	public static Set<Integer>[] I_u;
	public static Set<Index>[] S_u;

	public static HashSet<Integer> I;
	public static Set<Index> S_;
	//some statistics, start from index "1"
	public static int[] user_rating_number;
	public static double[] userRatingSumTrain;
	public static int[] userRatingNumTrain;

	public static int start_hybrid_averaging_iteration=50;//10
	public static int local_train_iterations = 50;
	public static double rho=1;
	
	public static HashMap<Integer,HashMap<Integer,Double>>[] rating;

}


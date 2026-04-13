package LLFT;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringTokenizer;
import java.util.Random;

public class ReadData {
	// private static final Random rand = new Random();
	public static  void readTrainData() throws IOException {
		File f=new File(Data.trainFilePath);
		BufferedReader Reader=new BufferedReader(new FileReader(f));
		String s=null;
		while((s=Reader.readLine())!=null) {
		
			StringTokenizer st=new StringTokenizer(s, "::");
			
			String i=null;
			if(st.hasMoreTokens())i=st.nextToken();
			
			String j=null;
			if(st.hasMoreTokens())j=st.nextToken();		
			
			String k=null;
			if(st.hasMoreTokens())k=st.nextToken();
			
			String value=null;
			if(st.hasMoreTokens())value=st.nextToken();
			
			//ת//////
			int a=Integer.valueOf(i);
			int b=Integer.parseInt(j);
			int c=Integer.parseInt(k);
			double val=Double.parseDouble(value);
//			System.out.println("i:"+a+" j:"+b+" k:"+c+" value:"+value+'\n');
			
			TensorTuple tt=new TensorTuple();
			tt.uid=a; tt.sid=b; tt.tid=c; tt.value=val;
		
			// Data.trainData1.add(tt);//存储方式1 所有训练样本的平面列表
			if(Data.trainData.containsKey(a)) { //按用户ID分组的映射（用户ID -> 该用户的所有评分）
				Data.trainData.get(a).add(tt);
			}
			else {
				ArrayList<TensorTuple> list=new ArrayList<>();
				list.add(tt);
				Data.trainData.put(a, list);
			}
			Data.trainCount++;
//			System.out.println("trainCount:"+Data.trainCount+'\n');
			
			//ͳ//һ// i,j ,k ,value////Ϣ//max value,minvlue
			Data.maxUid=Data.maxUid>a?Data.maxUid:a;
			Data.maxSid=Data.maxSid>b?Data.maxSid:b;
			Data.maxTid=Data.maxTid>c?Data.maxTid:c;
			Data.maxValue=Data.maxValue>val?Data.maxValue:val;
			
			Data.minUid=Data.minUid<a?Data.minUid:a;
			Data.minSid=Data.minSid<b?Data.minSid:b;
			Data.minTid=Data.minTid<c?Data.minTid:c;
			Data.minValue=Data.minValue<val?Data.minValue:val;
		    
		    int uid = tt.uid;
            int sid = tt.sid;
            int tid = tt.tid;//get uid sid tid
        }
//		System.out.println("maxUid:"+Data.maxUid+" maxSid:"+Data.maxSid+" maxTid:"+Data.maxTid+" maxValue:"+Data.maxValue);
//		System.out.println("minUid:"+Data.minUid+" minSid:"+Data.minSid+" minTid:"+Data.minTid+" minValue:"+Data.minValue);
		Reader.close();

		Data.userRatingNumTrain=new int[Data.maxUid+1];
		Data.userRatingSumTrain=new double[Data.maxUid+1];

		Data.S_=new HashSet<>();
		Data.S_u=new HashSet[Data.maxUid+1];
		Data.rating=new HashMap[Data.maxUid+1];

		for(int j=1;j<Data.maxSid+1;j++) {
			for(int k=1;k<Data.maxTid+1;k++) {
				Index t=new Index();
				t.j=j;
				t.k=k;
				Data.S_.add(t);
			}
		}
		
		for(int i=1;i<Data.maxUid+1;i++) {
			Data.S_u[i]=new HashSet<Index>();
		}
		
		

		for(Integer uid:Data.trainData.keySet()) {

			for(TensorTuple e:Data.trainData.get(uid)) {
				Data.userRatingNumTrain[uid]++;
				Data.userRatingSumTrain[uid]+=e.value;
				//Data.I_u[uid].add(e.sid);

				Index t=new Index();
				t.j=e.sid;
				t.k=e.tid;
				Data.S_u[uid].add(t);
				//initialize real rating
				if(Data.rating[uid].containsKey(e.sid)) {
					Data.rating[uid].get(e.sid).put(e.tid, e.value);
				}
				else {
					HashMap<Integer, Double> temp=new HashMap<>();
					temp.put(e.tid, e.value);
					Data.rating[uid].put(e.sid, temp);
				}
			}
		}

		Data.u_set=new HashSet<>(Data.trainData.keySet());
//		for(Integer e:Data.u_set) {
//			System.out.println("uid:"+e);
//		}
	}
	
	public static  void readTestData() throws IOException {
		File f=new File(Data.testFilePath);
		BufferedReader Reader=new BufferedReader(new FileReader(f));
		String s=null;
		while((s=Reader.readLine())!=null) {

			StringTokenizer st=new StringTokenizer(s, "::");
			String i=null;
			if(st.hasMoreTokens())i=st.nextToken();
			String j=null;
			if(st.hasMoreTokens())j=st.nextToken();		
			String k=null;
			if(st.hasMoreTokens())k=st.nextToken();
			String value=null;
			if(st.hasMoreTokens())value=st.nextToken();
			
			int a=Integer.valueOf(i);
			int b=Integer.parseInt(j);
			int c=Integer.parseInt(k);
			double val=Double.parseDouble(value);
//			System.out.println("i:"+a+" j:"+b+" k:"+c+" value:"+value+'\n');
		
			TensorTuple tt=new TensorTuple();
			tt.uid=a; tt.sid=b; tt.tid=c; tt.value=val;

			Data.testData.add(tt);
	
			Data.testCount++;
//			System.out.println("testCount:"+Data.testCount+'\n');
		}
	
		Reader.close();
	}
}

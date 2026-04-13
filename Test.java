package LLFT;

import java.io.FileWriter;
import java.io.IOException;

public class Test {
	public static void test(int round, FileWriter fw) throws IOException {
		// if(round == 1 || round % 2 == 0) {	
			double sum=0;
			double sum1=0;
			for(TensorTuple e:Data.testData) {
				double pre=0;
				for(int r=1;r<Data.rank+1;r++) {
					pre+=Data.U[e.uid][r]*Data.S[e.sid][r]*Data.T[e.tid][r];
				}
				pre=pre/(2*Data.rank)+0.5;
				pre=pre*(Data.maxValue-Data.minValue)+Data.minValue;
				
				double err=e.value-pre;
				sum1+=Math.abs(err);
				err=err*err;
				sum+=err;
			}
			double rmse=Math.sqrt(sum/Data.testCount);
			double mae=(sum1/Data.testCount);
			System.out.println("round:"+round+"\t"+rmse+'\t'+mae);
			fw.write("round"+round+": "+rmse+"\t"+mae+'\n');
			fw.flush();
			
			Data.lastTimeRMSE=Data.thisTimeRMSE;
			Data.thisTimeRMSE=rmse;
			
			Data.lastTimeMAE=Data.thisTimeMAE;
			Data.thisTimeMAE=mae;
			
			if(rmse<Data.minRMSE)Data.minRMSE=rmse;
			if(mae<Data.minMAE)Data.minMAE=mae;
	}
}

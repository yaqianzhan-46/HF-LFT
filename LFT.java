package LLFT;

import java.util.Random;

public class LFT {
	public static double eta=0.001;//0.001
	public static double lambda1=0.001;
	public static double minRMSE=10000;
	public static int threshold=5;
	public static double lastRMSE=10000;
	public static double currentRMSE=0;
	public static int count1=0;
	
	public static void initLatentMatrixWithBinaryCodeByTrainedFactorMatrix(int count) {
		
		Data.U=new double[Data.maxUid+1][Data.rank+1];
		Data.S=new double[Data.maxSid+1][Data.rank+1];
		Data.T=new double[Data.maxTid+1][Data.rank+1];
		
		
		double initScale=0.05;
		int scale=1000;
		Random ran=new Random();//
		for(int i=0;i<=Data.maxUid;i++) {
			for(int r=0;r<=Data.rank;r++) {
				Data.U[i][r]=(double)ran.nextInt(scale)/scale*initScale;			
			}
		}		
		for (int j = 0; j <= Data.maxSid; j++) {
			for (int r = 0; r <= Data.rank; r++) {
				Data.S[j][r] = (double)ran.nextInt(scale)/scale*initScale;
			}
		}
		for (int k = 0; k <= Data.maxTid; k++) {
			for (int r = 0; r <= Data.rank; r++) {
				Data.T[k][r] = (double)ran.nextInt(scale)/scale*initScale;
			}
		}

		threshold = 5;
		lastRMSE=10000;
		currentRMSE=0;
		for (int t = 0; t < count; t++) {
			// 
			for (TensorTuple e : Data.trainData1) {
				double pre = 0;
				for (int r = 1; r < Data.rank + 1; r++) {
					pre += Data.U[e.uid][r] * Data.S[e.sid][r] * Data.T[e.tid][r];
				}
//				double err = (e.value-Data.minValue)/(Data.maxValue-Data.minValue) - pre;// 
				double err=e.value-pre;

				//
				for (int r = 1; r < Data.rank + 1; r++) {
					double uir = -err * Data.S[e.sid][r] * Data.T[e.tid][r] + lambda1 * Data.U[e.uid][r];
					double sjr = -err * Data.U[e.uid][r] * Data.T[e.tid][r] + lambda1 * Data.S[e.sid][r];
					double tkr = -err * Data.U[e.uid][r] * Data.S[e.sid][r] + lambda1 * Data.T[e.tid][r];

					// //////
					Data.U[e.uid][r] = Data.U[e.uid][r] - eta * uir;
					Data.S[e.sid][r] = Data.S[e.sid][r] - eta * sjr;
					Data.T[e.tid][r] = Data.T[e.tid][r] - eta * tkr;
				}
			}
			// test
			double sum = 0;
			for (TensorTuple e : Data.testData) {
				double pre = 0;
				for (int r = 1; r < Data.rank + 1; r++) {
					pre += Data.U[e.uid][r] * Data.S[e.sid][r] * Data.T[e.tid][r];
				}
//				double err =  e.value - (pre*(Data.maxValue-Data.minValue)+Data.minValue);
				double err=e.value-pre;
				sum = sum + err * err;
			}
			double rmse = Math.sqrt(sum / Data.testCount);
			lastRMSE=currentRMSE;
			currentRMSE=rmse;
			System.out.println(t + "	rmse:" + rmse);
			
			if(rmse<minRMSE)minRMSE=rmse;
			
			if(lastRMSE-currentRMSE<0.0001) {
				count1++;
				if(count1==threshold)
					break;
			}
			else {
				count1=0;
			}
		}
//		 System.out.println("minRMSE:"+minRMSE);

		 ////ʼΪ ////1
//		for (int i = 0; i <= Data.maxUid; i++) {
//			for (int r = 0; r <= Data.rank; r++) {
//				System.out.print(Data.U[i][r] + " 	");
//			}
//			System.out.println();
//		}
//		for (int j = 0; j <= Data.maxSid; j++) {
//			for (int r = 0; r <= Data.rank; r++) {
//				System.out.print(Data.S[j][r] + "	 ");
//			}
//			System.out.println();
//		}
//		for (int k = 0; k <= Data.maxTid; k++) {
//			for (int r = 0; r <= Data.rank; r++) {
//				System.out.print(Data.T[k][r] + " 	");
//			}
//			System.out.println();
//		}

		/////Ϊ////һ//0.5Ϊ/ֽ///
		double line=0.5;
		double max=-1000;
		double min=1000;
		for (int i = 0; i <= Data.maxUid; i++) {
			for (int r = 0; r <= Data.rank; r++) {
				if(max<Data.U[i][r])max=Data.U[i][r];
				if(min>Data.U[i][r])min=Data.U[i][r];
			}
//			System.out.println();
		}
		
		line=(max+min)/2;
		for (int i = 0; i <= Data.maxUid; i++) {
			for (int r = 0; r <= Data.rank; r++) {
				if (Data.U[i][r] >= line) {
					Data.U[i][r] = 1;
				}
				else {
					Data.U[i][r]=-1;
				}
			}
		}
		
		max=-1000;min=1000;
		for (int j = 0; j <= Data.maxSid; j++) {
			for (int r = 0; r <= Data.rank; r++) {
				if(max<Data.S[j][r])max=Data.S[j][r];
				if(min>Data.S[j][r])min=Data.S[j][r];
			}
//			System.out.println();
		}
		line=(max+min)/2;
		for (int j = 0; j <= Data.maxSid; j++) {
			for (int r = 0; r <= Data.rank; r++) {
				if(Data.S[j][r]>=line) {
					Data.S[j][r]=1;
				}
				else {
					Data.S[j][r]=-1;
				}
			}
		}
		
		max=-1000;min=1000;
		for (int k = 0; k <= Data.maxTid; k++) {
			for (int r = 0; r <= Data.rank; r++) {
				if(max<Data.T[k][r])max=Data.T[k][r];
				if(min>Data.T[k][r])min=Data.T[k][r];
			}
//			System.out.println();
		}
		line=(max+min)/2;
		for (int k = 0; k <= Data.maxTid; k++) {
			for (int r = 0; r <= Data.rank; r++) {
				if(Data.T[k][r]>=line) {
					Data.T[k][r]=1;
				}
				else {
					Data.T[k][r]=-1;
				}
			}
		}
		
//		for (int i = 0; i <= Data.maxUid; i++) {
//			for (int r = 0; r <= Data.rank; r++) {
//				System.out.print(Data.U[i][r] + "	 ");
//			}
//			System.out.println();
//		}
//		for (int j = 0; j <= Data.maxSid; j++) {
//			for (int r = 0; r <= Data.rank; r++) {
//				System.out.print(Data.S[j][r] + " ");
//			}
//			System.out.println();
//		}
//		for (int k = 0; k <= Data.maxTid; k++) {
//			for (int r = 0; r <= Data.rank; r++) {
//				System.out.print(Data.T[k][r] + " 	");
//			}
//			System.out.println();
//		}
	}
}

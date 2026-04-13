package LLFT;

import java.io.IOException;
import java.util.Random;

public class Main1 {

	public static void main(String[] args) throws IOException {
		
		
		Data.trainFilePath="Dataset\\1\\CDs_tr.txt";
		Data.testFilePath="Dataset\\1\\CDs_ts.txt";
		
		// Data.trainFilePath="Dataset\\3\\Network1_tr.txt";
		// Data.testFilePath="Dataset\\3\\Network1_ts.txt";
		
		// Data.trainFilePath="Dataset\\2\\Network2_tr.txt";
		// Data.testFilePath="Dataset\\2\\Network2_ts.txt";
		
		// Data.trainFilePath="Dataset\\2\\TV_tr.txt";
		// Data.testFilePath="Dataset\\2\\TV_ts.txt";

		ReadData.readTrainData();
		ReadData.readTestData();
		System.out.println("[INFO] 数据维度：用户数=" + Data.maxUid + "，物品数=" + Data.maxSid + "，时间数=" + Data.maxTid);
		
		//2.
		Data.count1=1;
		Data.lambda=1e-8;
		Data.trainRound=100;
		Data.rank=64;
		Data.p=1; 
		Data.epsilon=0;
		//Data.initNoiseScale();//LDP
		
		//3		
		Main1.initMatrix();
		
		//lightFR 
		Main1.initGradientMtrixInServer();
		
		//4
		System.out.println("[INFO] 开始模型训练！总轮次：" + Data.trainRound);
		Server.train();
	}
	
	public static void initMatrix1() {
		// hash
		Data.U = new double[Data.maxUid + 1][Data.rank + 1];
		Data.S = new double[Data.maxSid + 1][Data.rank + 1];
		Data.T = new double[Data.maxTid + 1][Data.rank + 1];

		//
		Random ran = new Random();

		// init U S T
		for (int i = 0; i <= Data.maxUid; i++) {
			for (int r = 0; r <= Data.rank; r++) {
				Data.U[i][r]=1;
			}
		}
		for (int j = 0; j <= Data.maxSid; j++) {
			for (int r = 0; r <= Data.rank; r++) {			
				Data.S[j][r] =1;
			}
		}
		for (int k = 0; k <= Data.maxTid; k++) {
			for (int r = 0; r <= Data.rank; r++) {
				Data.T[k][r] =1;
			}
		}
	}
	
	public static void initMatrix() {
		// hash
		Data.U = new double[Data.maxUid + 1][Data.rank + 1];
		Data.S = new double[Data.maxSid + 1][Data.rank + 1];
		Data.T = new double[Data.maxTid + 1][Data.rank + 1];
		Data.dummy_U= new double[Data.maxUid+1][Data.rank+1]; //dummyU
		Data.dummy_S= new double[Data.maxSid+1][Data.rank+1]; //dummy_S

		//
		Random ran = new Random();// 

		// init U S T
		for (int i = 0; i <= Data.maxUid; i++) {
			for (int r = 0; r <= Data.rank; r++) {
				Data.U[i][r]= 1;
			}
		}
		for (int j = 0; j <= Data.maxSid; j++) {
			for (int r = 0; r <= Data.rank; r++) {
				Data.S[j][r] = 1;
			}
		}
		for (int k = 0; k <= Data.maxTid; k++) {
			for (int r = 0; r <= Data.rank; r++) {
				Data.T[k][r] = 1;
			}
		}

		for (int i = 0; i <= Data.maxUid; i++) {
			for (int r = 0; r <= Data.rank; r++) {
				Data.dummy_U[i][r]= 1;
			}
		}
	}
	
	public static void initGradientMtrixInServer() {
		Data.SGradient=new double[Data.maxSid+1][Data.rank+1];
		Data.TGradient=new double[Data.maxTid+1][Data.rank+1];
		
		for(int j=0;j<=Data.maxSid;j++) {
			for(int r=0;r<=Data.rank;r++) {
					Data.SGradient[j][r]=0;					
			}
		}
		
		for(int k=0;k<=Data.maxTid;k++) {
			for(int r=0;r<=Data.rank;r++) {
					Data.TGradient[k][r]=0;				
			}
		}
	}
}

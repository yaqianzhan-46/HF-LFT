package LLFT;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.Thread.State;
import java.util.ArrayList;
import java.util.Collections;
import java.util.function.Predicate;

//import org.omg.Messaging.SyncScopeHelper;

public class Server {
	
	public static void train() throws IOException {
		
		FileWriter fw=new FileWriter(Data.trainFilePath.replace(".txt", "_")+Data.rank+"_"+Data.lambda+"_"+Data.p+"_"+
		System.currentTimeMillis()/1000+"localcount_"+Data.count1+"_"+Data.init); 
		
//		FileWriter fw1=new FileWriter(Data.trainFilePath.replace(".txt", "_")+Data.rank+"_"+Data.lambda+"_"+Data.p+"_"+"_efficiency"+System.currentTimeMillis()/1000+"localcount_"+Data.count1+"_"+Data.init);
		
		fw.write("round\t everyRMSE\t everyMAE\n");
		fw.flush();

		
		long sumtime=0;
		for (int round = 1; round <= Data.trainRound; round++) {
			
			int num = (int) (Data.p * Data.maxUid);
			// System.out.println("\n[INFO] 第 " + round + "/" + Data.trainRound + " 轮训练开始...");
			ArrayList<Integer> I_c = new ArrayList<>(Data.u_set);

			Server.resetGradientMatrixInServer();//reset
			
			Collections.shuffle(I_c);
			
			for (int n = 0; n < num; n++) {		
				
				if(n>=I_c.size())continue;
				int uid = I_c.get(n);
				Client u = new Client(uid);
				// 
				u.train();

			
				ArrayList<GradientTuple> gradientFrom_u =Data.Gradient.get(uid);
				for (GradientTuple e : gradientFrom_u) {
					// 
					int sid = e.sid;
					int tid = e.tid;
					
					for (int r = 1; r < Data.rank + 1; r++) {
						Data.SGradient[sid][r] += e.gradient_Sj.get(r - 1);
						Data.TGradient[tid][r] += e.gradient_Tk.get(r - 1);
					}
				}
				Data.Gradient.clear();
			}
		
			// S T
			for (int j = 1; j < Data.maxSid + 1; j++) {
				for (int r = 1; r < Data.rank + 1; r++) {
					if (Data.SGradient[j][r] > 0) {
						Data.S[j][r] = 1;
					}
					if (Data.SGradient[j][r] < 0)
						Data.S[j][r] = -1;
				}
			}
			for (int j = 1; j < Data.maxTid + 1; j++) {
				for (int r = 1; r < Data.rank + 1; r++) {
					if (Data.TGradient[j][r] > 0) {
						Data.T[j][r] = 1;
					}
					if (Data.TGradient[j][r] < 0)
						Data.T[j][r] = -1;
				}
			}
			// System.out.println("[INFO] 第 " + round + " 轮训练结束，开始测试...");
			Test.test(round, fw);
			// System.out.println("[INFO] 第 " + round + " 轮测试结果：RMSE=" + Data.thisTimeRMSE + "，MAE=" + Data.thisTimeMAE);
			
			if(Data.lastTimeRMSE-Data.thisTimeRMSE<0.001) {
				Data.count++;
				if(Data.count==Data.threshold)break;
			}
			else {
				Data.count=0;
			}
//			System.out.println(rmse);
		}
	
		System.out.println("minRMSE:"+Data.minRMSE+"\t"+"minMAE:"+Data.minMAE);
		fw.write("minRMSE:"+Data.minRMSE+"\t"+"minMAE:"+Data.minMAE);
		fw.flush();
		fw.close();	
//		fw1.close();
	}

	private static void resetGradientMatrixInServer() {
		// 
		for(int j=1;j<Data.maxSid+1;j++) {
			for(int r=1;r<Data.rank+1;r++) {
				Data.SGradient[j][r]=0;
			}
		}
		for(int k=1;k<Data.maxTid+1;k++) {
			for(int r=1;r<Data.rank+1;r++) {
				Data.TGradient[k][r]=0;
			}
		}
	}
}

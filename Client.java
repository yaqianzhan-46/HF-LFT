package LLFT;

import java.util.ArrayList;

public class Client {
	public int uid;
	public double lambda=0;
	public int localTrainRound=20;
	public ArrayList<TensorTuple> mydata=null;

	public Client(int id) {
	
		uid=id;
		lambda=Data.lambda;
		localTrainRound=(int) (Data.trainRound*0.2);
		mydata=Data.trainData.get(id);
	}

	public void train() {
	//update U
		int flag=0;
		double[] ui=new double[Data.rank+1];
		for(int r=0;r<Data.rank+1;r++) {
			ui[r]=Data.U[uid][r];
		}
		
		int count=0;
		while(count<Data.count1) {
			flag=0;
			
			count++;
			
			for(int r=1;r<Data.rank+1;r++) {
				for(int n=1;n<Data.rank+1;n++) {
					ui[n]=Data.U[uid][n];
				}
				ui[r]=0;

				double uir_hat=0;
				for(TensorTuple e:mydata) {
					int sid=e.sid;
					int tid=e.tid;
					
					double rest=0;
					for(int n=1;n<Data.rank+1;n++) {
						rest=rest+ui[n]*Data.S[sid][n]*Data.T[tid][n];
					}
					rest=rest/(2*Data.rank)+0.5;
					uir_hat=uir_hat+1.0/Data.rank*((e.value-Data.minValue)/(Data.maxValue-Data.minValue)-rest)*Data.S[sid][r]*Data.T[tid][r];
					
					double reg=0;
					for(int i=1;i<Data.rank+1;i++) {
						reg+=ui[i];
					}
					reg=reg*2*lambda;
					uir_hat=uir_hat-reg;
				}
			
//				double reg=0;
//				for(int i=1;i<Data.rank+1;i++) {
//					reg+=ui[i];
//				}
//				reg=reg*2*lambda;
//				uir_hat=uir_hat-reg;
				
				double uir_new=0;
				if(uir_hat>0)uir_new=1;
				else if(uir_hat<0)uir_new=-1;
				else uir_new=Data.U[uid][r];
				
				if(Data.U[uid][r]!=uir_new) {
					flag=1;
					Data.U[uid][r]=uir_new;
				}
//				System.out.print("r="+r+":"+Data.U[uid][r]+"\t");
//				System.out.println("r="+r+"\t"+flag);
			}
			
//			System.out.println();
//			if(flag==0)break;	
		}
		
		
		ArrayList<GradientTuple> Gra=new ArrayList<>();
		for(TensorTuple e:mydata) {
		
			int sid=e.sid;
			int tid=e.tid;
			double[] sj=new double[Data.rank+1];
			double[] tk=new double[Data.rank+1];
			for(int r=0;r<Data.rank+1;r++) {
				sj[r]=Data.S[sid][r];
				tk[r]=Data.T[tid][r];
			}
			
			ArrayList<Double> grasj=new ArrayList<>(Data.rank);
			ArrayList<Double> gratk=new ArrayList<>(Data.rank);
			for(int r=1;r<Data.rank+1;r++) {
				//rest
				for(int n=1;n<Data.rank+1;n++) {
					sj[n]=Data.S[sid][n];
					tk[n]=Data.T[tid][n];
				}
				sj[r]=0;
				tk[r]=0;
				double rest=0;
				double rest1=0;
				for(int n=1;n<Data.rank+1;n++) {
					rest+=sj[n]*Data.U[uid][n]*Data.T[tid][n];
					rest1+=tk[n]*Data.U[uid][n]*Data.S[sid][n];
				}
				rest=rest/(2*Data.rank)+0.5;
				rest1=rest1/(2*Data.rank)+0.5;
				double graSjr=((e.value-Data.minValue)/(Data.maxValue-Data.minValue)-rest)*Data.U[uid][r]*Data.T[tid][r];
				double graTkr=((e.value-Data.minValue)/(Data.maxValue-Data.minValue)-rest1)*Data.U[uid][r]*Data.S[sid][r];
				
				grasj.add(r-1, graSjr);
				gratk.add(r-1, graTkr);
			}
			
			GradientTuple g=new GradientTuple();
			g.sid=sid;g.tid=tid;
			//boolean flipped = false;
			for(int r=1;r<=Data.rank;r++) {
				//rest
				double rest=0,rest1=0;
				for(int n=1;n<=Data.rank;n++) {
					if(n!=r) {
						rest=rest+Data.S[sid][n];
						rest1=rest1+Data.T[tid][n];
					}
				}
				double a=grasj.get(r-1);
				a=a/Data.rank-2*lambda*rest;
				if(a>0)a=1;
				else if(a<0) {a=-1;}
				else {a=Data.S[sid][r];}
				grasj.set(r-1, a);
				
				a=gratk.get(r-1);
				a=a/Data.rank-2*lambda*rest1;
				if(a>0)a=1;
				else if(a<0) {a=-1;}
				else {a=Data.T[tid][r];}
				gratk.set(r-1, a);
			}
			
			g.gradient_Sj=grasj;g.gradient_Tk=gratk;

			Gra.add(g);
		}
//		for(GradientTuple e:Gra) {
//			System.out.println("sid:"+e.sid+"	tid:"+e.tid);
//			for(int r=0;r<Data.rank;r++) {
//				System.out.print(e.gradient_Sj.get(r)+"\t");
//			}
//			System.out.println();
//			for(int r=0;r<Data.rank;r++) {
//				System.out.print(e.gradient_Tk.get(r)+"\t");
//			}
//			System.out.println();
//		}

		Data.Gradient.put(uid, Gra);
	}
}

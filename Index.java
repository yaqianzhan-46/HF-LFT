package LLFT;

import java.util.Objects;

public class Index {
	public int j;
	public int k;
	
	@Override
	public boolean equals(Object obj) {
		if(obj==this) {
			return true;
		}
		if(obj==null || obj.getClass()!=this.getClass()) {
			return false;
		}
		Index in=(Index)obj;

		return this.j==in.j && this.k==in.k;
	}

	@Override
	public int hashCode() {		
		return Objects.hash(j,k);
	}
}


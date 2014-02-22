import numpy as np
from scipy import io as sio
def perform_lda(X, K, alpha, beta, num_epochs):
	it = 0  #iterations
	(M, V) = X.shape
	Z = np.random.randint(3, size=X.nonzero()[0].size) #Z is defined for the whole document
	(q,sum_q) = load_q(X,K,Z)
	(n, sum_n) = load_n(X,K,Z)
	sum_alpha = np.sum(alpha)
	sum_beta = np.sum(beta)
	p = np.empty(K, dtype=float)
	while(it<num_epochs):
		i = 0 #denotes the position of the word
		for m in range(2):
			nz =  X[m,:].nonzero()[1]
			for w_i in ([nz[temp] for temp in range(nz.size) for temp2 in range(int(X[m,nz[temp]]))]):
				for j in range(K):
					p[j] = (q[j,w_i]+beta[w_i])/(sum_q[j]+sum_beta) * (n[m,j]+alpha[j])	/(sum_n[m]+sum_alpha)
					# q(.) and n(.) should be changed to q' and n' appropriately
				p =  p/np.sum(p)
				#randomly sample and update Z[i] based on p
				#update q,p,sum_p, sum_q appropriately
				i+=1  #increment i with every occurance of word
				
		it+=1	
	return 0


def load_q (X, K, V, Z, M):
	q = np.zeros((K,V), dtype=int)
	for m in range(M):
		nz =  X[m,:].nonzero()[1]
		for w_i in ([nz[temp] for temp in range(nz.size) for temp2 in range(int(X[m,nz[temp]]))]):
		#actually load_n and load_q can be merged in this single big loop
			d = 0 #arbit delete this line 
		
	return q

def load_n(X, K, M, Z):
	n = np.zeros((M,K), dtype=int)
	return n
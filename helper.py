import numpy as np
from scipy import io as sio
def perform_lda(X, K, alpha, beta, num_epochs):
	print "LDA started...."
	it = 0  #iterations
	(M, V) = X.shape
	Z = np.random.randint(K, size=X.sum()) #Z is defined for the whole document
	(q,n) = load(X,K,V,Z,M)
	sum_q = np.sum(q, axis=0)
	sum_n = np.sum(n, axis=1)
	sum_alpha = np.sum(alpha)
	sum_beta = np.sum(beta)
	p = np.empty(K, dtype=float)
	while(it<num_epochs):
		print "Iteration #" ,it+1, " of ", num_epochs
		i = 0 #denotes the position of the word
		for m in range(M):
			nz =  X[m,:].nonzero()[1]
			for w_i in ([nz[temp] for temp in range(nz.size) for temp2 in range(int(X[m,nz[temp]]))]):
				for j in range(K):
					err = -1 if Z[i]==j else 0
					q_bar,n_bar = q[j,w_i] + err, n[m,j] + err
					p[j] = (q_bar+beta[w_i])/(sum_q[j]+err+sum_beta) * (n_bar+alpha[j])	/(sum_n[m]+err+sum_alpha)
				p =  p/(np.sum(p))
				pre = Z[i]
				cur = Z[i] = int(np.random.choice(K,1,p=p)) #randomly sample and update Z[i] based on p
				#update q, sum_q, n, sum_n appropriately 
				q[pre,w_i], q[cur,w_i] = q[pre,w_i]-1, q[cur,w_i]+1
				sum_q[pre],sum_q[cur] = sum_q[pre]-1,sum_q[cur]+1
				n[m,pre], n[m,cur] = n[m,pre]-1, n[m,cur]+1  
				i+=1  #increment i with every occurance of word
		it+=1
	print "finished ..."
	return Z


def load (X, K, V, Z, M):
	q = np.zeros((K,V), dtype=int)
	n = np.zeros((M,K), dtype=int)
	i=0
	for m in range(M):
		nz =  X[m,:].nonzero()[1]
		for w_i in ([nz[temp] for temp in range(nz.size) for temp2 in range(int(X[m,nz[temp]]))]):
			q[Z[i],w_i]+=1
			n[m,Z[i]]+=1
			i+=1
	return (q,n)

def predict(Z, K, doc_len):
	print doc_len.shape
	strt = 0
	count = np.empty(K, dtype=int)
	y = np.empty(doc_len.size, dtype=int)
	for i in range(doc_len.size):
		end = strt + int(doc_len[i])-1;
		for j in range(K):
			count[j] = np.sum(Z[strt:end] == j)
		y[i] = np.argmax(count) + 1
		strt = end+1
	return y
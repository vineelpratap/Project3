import numpy as np
from scipy import io as sio

def perform_lda(X, K, M, V, alpha, beta, num_epochs):
	
############ initialisation ###########################################	
	
	Z = [] #topics of words of documents
	q = np.zeros((K,V))+ beta #numerator term
	n = np.zeros((M,K))+ alpha #numerator term
	doc = []
	
	for m in range(M):
		Z_doc = np.random.randint(K, size=X[m,:].sum())		#can do smart initialisation
		nz =  X[m,:].nonzero()[1]
		doc.append([nz[temp] for temp in range(nz.size) for temp2 in range(int(X[m,nz[temp]]))])
		for i, w_i in enumerate(doc[m]):
			z = Z_doc[i]
			q[z,w_i],n[m,z] = q[z,w_i]+1,n[m,z]+1
		Z.append(Z_doc)
	Q = np.sum(q,axis=1) + np.sum(beta) #denominator term
	p = np.empty(K, dtype=float)


############ inference #############################################	
	 
	it = 0  #iterations
	while(it < num_epochs):
		
		print "epoch #" ,it+1, " of ", num_epochs
		for m in range(M):
			for i, w_i in enumerate(doc[m]):
				
				#discount for i-th word w_i with topic z
				z = Z[m][i]
				q[z,w_i], n[m,z], Q[z] = q[z,w_i]-1, n[m,z]-1, Q[z]-1
				
				# sampling topic new_z for w_i
				p = q[:,w_i] * n[m,:]  / Q
				new_z = np.random.multinomial(1, p / p.sum()).argmax()
				
				# set z the new topic and increment counters
				Z[m][i] = new_z
				q[new_z,w_i], n[m,new_z], Q[new_z] = q[new_z,w_i]+1, n[m,new_z]+1, Q[new_z]+1
				
		it+=1
	print "lda finished ..."
	return Z


def predict(Z, K, M):
	count = np.empty(K, dtype=int)
	y = np.empty(len(Z), dtype=int)
	compare = [1..K]
	for m in range(M):
		for j in range(K):
			count[j] = np.sum(Z[m] == j)
		y[m] = np.argmax(count) + 1
	return y
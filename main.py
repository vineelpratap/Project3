from helper import *
f = sio.loadmat('classic400.mat', squeeze_me=True, struct_as_record=False)
#[('classic400', (400, 6205), 'sparse'), ('truelabels', (1, 400), 'double'), ('classicwordlist', (6205, 1), 'cell')]
vocab =  f['classicwordlist']
Y = np.array(f['truelabels'])
X = f['classic400'] #scipy.sparse.csc_matrix


K,(M,V) = 3,X.shape #no. of topics
a = (20.0/K)*np.ones(K) #alpha - parameter of topics prio
b = 0.1*np.ones(V) #beta -  parameter of words prior

Z = perform_lda(X, K, M, V, a, b, 2000, vocab, 10) #infers z
Y_pred = predict(Z, K, M)
np.savetxt("y.txt",Y, fmt='%d')
np.savetxt("y_pred.txt", Y_pred, fmt='%d')

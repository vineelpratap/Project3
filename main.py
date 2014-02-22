from helper import *
f = sio.loadmat('classic400.mat', squeeze_me=True, struct_as_record=False)
#[('classic400', (400, 6205), 'sparse'), ('truelabels', (1, 400), 'double'), ('classicwordlist', (6205, 1), 'cell')]
vocab =  f['classicwordlist']
Y = np.array(f['truelabels'])
X = f['classic400'] #scipy.sparse.csc_matrix
K = 3 #no. of topics
a = np.ones(K, dtype=float) #alpha
b = np.ones(len(vocab), dtype=float) #beta
Z = perform_lda(X, K, a, b, 30) #infers z
Y_pred = predict(Z, K, X.sum(axis=1))
np.savetxt("y.txt",Y, fmt='%d')
np.savetxt("y_pred.txt", Y_pred, fmt='%d')
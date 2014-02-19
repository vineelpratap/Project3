from helper import *
f = sio.loadmat('classic400.mat', squeeze_me=True, struct_as_record=False)
#[('classic400', (400, 6205), 'sparse'), ('truelabels', (1, 400), 'double'), ('classicwordlist', (6205, 1), 'cell')]
vocab =  f['classicwordlist']
Y = np.array(f['truelabels'])
X = f['classic400'] #scipy.sparse.csc_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


# -----------------------------------------------------------------------

def pca(X):
    # normalize the features
    X=(X - X.mean()) / X.std()

    # compute the covariance matrix
    X=np.matrix(X)
    cov=(X.T * X) / X.shape[0]
    #    print('cov \n', cov)
    #    print()
    # perform SVD
    U,S,V=np.linalg.svd(cov)  # singular value decomposition

    return U,S,V


def project_data(X,U,k):
    U_reduced=U[:,:k]
    return np.dot(X,U_reduced)


def recover_data(Z,U,k):
    U_reduced=U[:,:k]
    return np.dot(Z,U_reduced.T)


# -----------------------------------
# Apply PCA

data = loadmat('./ex7data1.mat')
X = data['X']
print(X.shape)
print(X)
print()

fig, ax = plt.subplots(figsize=(9,6))
ax.scatter(X[:, 0], X[:, 1])


U, S, V = pca(X)
print(U)
print()
print(S)
print()
print(V)



Z = project_data(X, U, 1)
print(Z)


X_recovered = recover_data(Z, U, 1)
print(X_recovered)
print(X_recovered.shape)

# -----------------------------------------------------------------------

# Apply PCA on faces

faces = loadmat('./ex7faces.mat')
X = faces['X']
print(X.shape)
plt.imshow(X)


# show one face
face = np.reshape(X[41,:], (32, 32))
plt.imshow(face)
plt.show()


U, S, V = pca(X)
Z = project_data(X, U, 100)

X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[41,:], (32, 32))
plt.imshow(face)

plt.show()


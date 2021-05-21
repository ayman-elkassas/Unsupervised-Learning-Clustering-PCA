import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


# -----------------------------------------------------------------------

#  select random points
def init_centroids(X,k):
    m,n=X.shape
    centroids=np.zeros((k,n))

    # get random index
    idx=np.random.randint(0,m,k)

    for i in range(k):
        centroids[i,:]=X[idx[i],:]

    return centroids


# centroid function
def find_closest_centroids(X,centroids):
    m=X.shape[0]
    k=centroids.shape[0]
    idx=np.zeros(m)

    for i in range(m):
        min_dist=1000000
        for j in range(k):
            dist=np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist=dist
                idx[i]=j

    return idx


## centroid maker
def compute_centroids(X,idx,k):
    m,n=X.shape
    centroids=np.zeros((k,n))

    for i in range(k):
        indices=np.where(idx == i)
        centroids[i,:]=(np.sum(X[indices,:],axis=1) / len(indices[0])).ravel()

    return centroids


# k means function
def run_k_means(X,initial_centroids,max_iters):
    m,n=X.shape
    k=initial_centroids.shape[0]
    idx=np.zeros(m)
    centroids=initial_centroids

    for i in range(max_iters):
        idx=find_closest_centroids(X,centroids)
        centroids=compute_centroids(X,idx,k)

    return idx,centroids


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

# we need to compress the image by express each cluster with mean by k-mean idea

image_data = loadmat('./bird_small.mat')

# print(image_data)

A = image_data['A']
# print(A.shape)
plt.imshow(A)
plt.show()

# normalize value ranges
A = A / 255

# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
# print(X.shape)

# randomly initialize the centroids
initial_centroids = init_centroids(X, 16)
# print(initial_centroids)


# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)

# get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int),:]

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

plt.imshow(X_recovered)

plt.show()
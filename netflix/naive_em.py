"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    
    mu , var , p = mixture
    d = X.shape[1]
    n = X.shape[0]
    K, _ = mixture.mu.shape
    post = np.zeros([n,K])
    post2 = np.zeros([n])
    ll = 0
    for i in range(n):
        tiled_vector = np.tile(X[i, :], (K, 1))
        logsse = np.log(p) + np.log((1/(2*np.pi*var)**(d/2))* np.exp((-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var)))
        sse = p*(1/(2*np.pi*var)**(d/2))* np.exp((-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
        sum_logsse = np.sum(np.exp(logsse))
        post[i,:] = np.exp(logsse- np.log(sum_logsse)  ) 
        #ll = np.sum(np.log(p)+np.log((1/(2*np.pi*var)**(d/2))) + (-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
        #lls = ll + lls
        post2[i] = np.log(np.sum(sse))

    ll = np.sum(post2)
    return post ,ll 
    
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
   
    d = X.shape[1]
    n = X.shape[0]
    K = post.shape[1]
    mu  = np.zeros([K, d])
    var = np.zeros([K])
    p = np.zeros([K])
    #var0 = np.zeros([n,d])
    var1 = np.zeros([n])

    for j in range(K):
        tiled_vector1 = np.tile(post[:, j], d)
        tiled_vector = np.reshape(tiled_vector1, (n, d), order = 'F')
        mu[j, :] = np.sum(tiled_vector*X,  axis=0)/np.sum(post[:, j])
        p[j] = np.sum(post[:, j])/n
    
    
    #print(mu[j, :])
        tiled_vector_mu1 = np.tile(mu[ j,:], n)
    #var0[j, :] =  ((X - mu[j,:])**2)
        tiled_vector_mu1 = np.reshape(   tiled_vector_mu1, (n, d), order = 'C')
    #print(tiled_vector_mu1)
        #var0 =  ((X - tiled_vector_mu1)**2)
    #print(var0)
        #var1 = np.sum(var0 , axis =1)
        #var[j] = np.sum(var1*post[:, j])/np.sum(post[:, j])
        var1 = (np.linalg.norm((X - tiled_vector_mu1), axis = 1)**2)
        var[j] = np.sum(post[:, j].reshape((1, n ))*var1)/(np.sum(post[:, j])*d)
        
        
    return GaussianMixture(mu, var, p) 
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    prev_ll = None
    ll = None
    while (prev_ll is None or ll - prev_ll > (10**(-6))*np.absolute(ll)):
        prev_ll = ll
        post ,ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll
    raise NotImplementedError

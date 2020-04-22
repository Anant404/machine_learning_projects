"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
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
    filter1 = np.zeros([K , d])
    
    ll = 0
    for i in range(n):
        tiled_vector = np.tile(X[i, :], (K, 1))
        filter1 = np.where(tiled_vector != 0, 1,0)
        filter2 = filter1[0]
        logsse = np.log(p + 1e-16) + (np.sum(filter2)/2)*np.log(1/(2*np.pi*var))+ (-(np.linalg.norm(tiled_vector- mu*filter1, axis = 1) )**2)/(2*var)
        
        sse = np.exp(logsse)
        sum_logsse = logsumexp(logsse)
        #post[i,:] = np.exp(logsse- np.log(sum_logsse)  )
        gf = logsse- sum_logsse
        post[i,:] = np.exp( gf ) 
        #ll = np.sum(np.log(p)+np.log((1/(2*np.pi*var)**(d/2))) + (-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
       
        post2[i] = sum_logsse

    ll = np.sum(post2)    
    return post ,ll
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    d = X.shape[1]
    n = X.shape[0]
    K = post.shape[1]
    mu  ,var , p1 = mixture
    p = np.zeros([K])
    var0 = np.zeros([n,d])
    var1 = np.zeros([n])
    filter1 = np.zeros([n,d])   

    for j in range(K):
        filter1 = np.where(X != 0, 1,0)
        tiled_vector1 = np.tile(post[:, j], d)
        tiled_vector = np.reshape(tiled_vector1, (n, d), order = 'F')
        for r in range(d):
            if np.sum(filter1[:, r]*post[:, j]) >= 1 :
                mu[j, r] = np.sum((tiled_vector[:, r]*X[:, r]),  axis=0)/np.sum(filter1[:, r]*post[:, j])       
        p[j] = np.sum(post[:, j])/n
        
        tiled_vector_mu1 = np.tile(mu[ j,:], n)
    #var0[j, :] =  ((X - mu[j,:])**2)
        tiled_vector_mu1 = np.reshape(   tiled_vector_mu1, (n, d), order = 'C')
        tiled_vector_mu2 = tiled_vector_mu1*filter1
    #print(tiled_vector_mu1)
    #var0 =  ((X - tiled_vector_mu1)**2)
    
    #print(var0)
    #var1 = np.sum(var0 , axis =1)
    
        var1 = (np.linalg.norm((X - tiled_vector_mu2), axis = 1)**2)
        var[j] = np.sum(post[:, j].reshape((1, n ))*var1)/(np.sum(post[:, j]*np.sum(filter1, axis = 1)))
       
    for v in range(K):
        if var[v] <= .25:
            var[v] = .25
            
        
        
    return GaussianMixture(mu  ,var , p)
    
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
        mixture = mstep(X, post,mixture, min_variance = .25)

    return mixture, post, ll
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mu , var , p = mixture
    d = X.shape[1]
    n = X.shape[0]
    K, _ = mixture.mu.shape
    post = np.zeros([n,K])
    post2 = np.zeros([n])
    filter1 = np.zeros([K , d])
    
    
    for i in range(n):
        tiled_vector = np.tile(X[i, :], (K, 1))
        filter1 = np.where(tiled_vector != 0, 1,0)
        filter2 = filter1[0]
        logsse = np.log(p + 1e-16) + (np.sum(filter2)/2)*np.log(1/(2*np.pi*var))+ (-(np.linalg.norm(tiled_vector- mu*filter1, axis = 1) )**2)/(2*var)
        
        sse = np.exp(logsse)
        sum_logsse = logsumexp(logsse)
        #post[i,:] = np.exp(logsse- np.log(sum_logsse)  )
        gf = logsse- sum_logsse
        post[i,:] = np.exp( gf ) 
        #ll = np.sum(np.log(p)+np.log((1/(2*np.pi*var)**(d/2))) + (-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
       
        post2[i] = sum_logsse
      
    # predictig vales
    
    d = X.shape[1]
    n = X.shape[0]
    K, _ = mu.shape
    pred = np.zeros([n,d])

    for i in range(n):
        #P1 = post[i, :]
        tiled_vector11 = np.tile(post[i,: ], d)
        tiled_vector22 = np.reshape(tiled_vector11, (K, d), order = 'F')
        
        pred[i, :] = np.sum(tiled_vector22*mu, axis = 0)
   
    
    
   
    filter22 = np.where(X != 0,0 ,1) 

    Xf = X + pred* filter22
        
   
    
    return Xf
    raise NotImplementedError

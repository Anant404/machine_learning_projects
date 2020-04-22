# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:29:40 2019

@author: admin
"""
import numpy as np
from scipy.special import logsumexp
   # mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    #var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    #p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component
X = np.array([[1,0,3], [0,0,0],[4,2,3],[2,1,3]])

mu , var , p = np.array([[2,1,3],[1,0, 3]]),np.array([4,5]), np.array( [.5, .4])
#mu , var , p = mixture
d = X.shape[1]
n = X.shape[0]
K, _ = mu.shape
post = np.zeros([n,K])
post2 = np.zeros([n])
ll = 0
for i in range(n):
    tiled_vector = np.tile(X[i, :], (K, 1))
    filter1 = np.where(tiled_vector != 0, 1,0)
    logsse = np.log(p) + np.log((1/(2*np.pi*var)**(np.sum(filter1[0])/2))* np.exp((-(np.linalg.norm(tiled_vector- mu*filter1, axis = 1) )**2)/(2*var)))
    sse = np.exp(logsse)
    sum_logsse = logsumexp(logsse)
        #post[i,:] = np.exp(logsse- np.log(sum_logsse)  )
    gf = logsse- sum_logsse
    post[i,:] = np.exp( gf ) 
        #ll = np.sum(np.log(p)+np.log((1/(2*np.pi*var)**(d/2))) + (-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
       
    post2[i] = np.log(np.sum(sse)) - sum_logsse

    

ll = np.sum(post2)
    
print( ll) 
        
    

    
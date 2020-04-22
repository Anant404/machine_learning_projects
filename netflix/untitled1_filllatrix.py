# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:29:40 2019

@author: admin
"""
import numpy as np
   # mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    #var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    #p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component
X = np.array([[1,0,3], [1,0,3],[4,2,3],[0,1,3]])
post = np.array([[.5,.3], [.1,.4],[.2,.4],[.3,.5]])
mu , var , p = np.array([[2,1,3],[1,0, 3]]),np.array([4,5]), np.array( [.5, .4])
d = X.shape[1]
n = X.shape[0]
K, _ = mu.shape
pred = np.zeros([n,d])

for i in range(n):
    P1 = post[i, :]
    tiled_vector1 = np.tile(post[i,: ], d)
    tiled_vector = np.reshape(tiled_vector1, (K, d), order = 'F')
    
    pred[i, :] = np.sum(tiled_vector*mu, axis = 0)
   
    
    
   
filter1 = np.where(X != 0,0 ,1) 

Xf = X + pred* filter1 
print(Xf) 
#d = X.shape[1]
#n = X.shape[0]
#K, _ = mu.shape
#ll = 0
#p1 = 0
#lls = 0
#p2 = 0
#p3 = 0
#post = np.zeros([n,K])
#post2 = np.zeros([n,K])
#for i in range(n):
#    tiled_vector = np.tile(X[i, :], (K, 1))
#    sse = (1/(2*np.pi*var)**(d/2))* np.exp((-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
#    sum_sse = np.sum(p*sse)
#    post[i,:] = p*sse/sum_sse
#    #ll = np.sum(np.log(p)+np.log((1/(2*np.pi*var)**(d/2))) + (-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
#    post2[i,:] = post[i,:]*(np.log(np.sum(p*sse))-np.log(post[i,:]))    
#    
#    
#    
#    for j in range(K):
#        #print(np.log(sum_sse)-np.log(post[i,j]))
#        p2 = post[i,j]*(np.log(sum_sse)-np.log(post[i,j])  )
#        p1 = p2 + p1
#    lls = p1 + lls
#    p2 = 0 
#    p1 = 0 
#        
#        
#    
#
#print(np.sum(post2))
#    
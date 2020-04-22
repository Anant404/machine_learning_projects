# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:29:40 2019

@author: admin
"""
import numpy as np
   # mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    #var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    #p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component
   #mu , var , p = np.array([[2,1,3],[1,0, 3]]),np.array([4,5]), np.array( [.5, .4])
   #Var: [0.05218451 0.06230449 0.03538519 0.05174859 0.04524244 0.05831186]


X = np.array([[0.85794562, 0.84725174],
 [0.6235637 , 0.38438171],
 [0.29753461, 0.05671298],
 [0.27265629, 0.47766512],
 [0.81216873, 0.47997717],
 [0.3927848 , 0.83607876],
 [0.33739616, 0.64817187],
 [0.36824154, 0.95715516],
 [0.14035078, 0.87008726],
 [0.47360805, 0.80091075],
 [0.52047748, 0.67887953],
 [0.72063265, 0.58201979],
 [0.53737323, 0.75861562],
 [0.10590761, 0.47360042],
 [0.18633234, 0.73691818]])

post = np.array([[0.15765074, 0.20544344, 0.17314824, 0.15652173, 0.12169798, 0.18553787],
 [0.1094766 , 0.22310587, 0.24109142 ,0.0959303 , 0.19807563, 0.13232018],
 [0.22679645, 0.36955206, 0.02836173 ,0.03478709, 0.00807236, 0.33243031],
 [0.16670188, 0.18637975, 0.20964608 ,0.17120102, 0.09886116, 0.16721011],
 [0.04250305, 0.22996176, 0.05151538 ,0.33947585, 0.18753121, 0.14901275],
 [0.09799086, 0.28677458, 0.16895715 ,0.21054678, 0.0069597 , 0.22877093],
 [0.16764519, 0.16897033, 0.25848053 ,0.18674186, 0.09846462, 0.11969746],
 [0.28655211, 0.02473762, 0.27387452 ,0.27546459, 0.08641467, 0.05295649],
 [0.11353057, 0.13090863, 0.20522811 ,0.15786368, 0.35574052, 0.03672849],
 [0.10510461, 0.08116927, 0.3286373  ,0.12745369, 0.23464272, 0.12299241],
 [0.09757735, 0.06774952, 0.40286261, 0.08481828, 0.1206645 , 0.22632773],
 [0.24899344, 0.02944918, 0.25413459, 0.02914503, 0.29614373, 0.14213403],  
 [0.35350682, 0.21890411, 0.26755234, 0.01418274, 0.10235276, 0.04350123],
 [0.15555757,  0.06236572, 0.16703133, 0.21760554, 0.03369562, 0.36374421],
 [0.1917808,  0.08982788, 0.17710673 ,0.03179658, 0.19494387, 0.31454414]])




#X = np.array([[1,2,3], [1,9,3],[4,2,3],[2,1,3]])
#post= np.array([[5,2,1], [2,9,1],[1,2,1],[2,3,1]])

d = X.shape[1]
n = X.shape[0]
K = post.shape[1]
mu  = np.zeros([K, d])
var = np.zeros([K])
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
        
    
    #print(mu[j, :])
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
    
#    sse = (1/(2*np.pi*var)**(d/2))* np.exp((-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
#    sum_sse = np.sum(p*sse)
#    post[i,:] = p*sse/sum_sse
#    #ll = np.sum(np.log(p)+np.log((1/(2*np.pi*var)**(d/2))) + (-(np.linalg.norm(tiled_vector- mu, axis = 1) )**2)/(2*var))
#    post2[i,:] = post[i,:]*(np.log(np.sum(p*sse))-np.log(post[i,:]))    
#    
    #print(var1)
#    
#    for j in range(K):
#        #print(np.log(sum_sse)-np.log(post[i,j]))
#        p2 = post[i,j]*(np.log(sum_sse)-np.log(post[i,j])  )
#        p1 = p2 + p1
#    lls = p1 + lls
#    p2 = 0 
#    p1 = 0 
#return GaussianMixture(mu, var, p)        
#        
#    
#
print(var)
#    
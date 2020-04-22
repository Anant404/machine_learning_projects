import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')

# TODO: Your code here
#for i in range(5):
mix, post   = common.init(X, 12,1)
    
mix1, post1, ll = em.run(X, mix, post)
X_pred = em.fill_matrix(X, mix1)

rse = common.rmse(X_gold, X_pred)
    #common.plot(X, mix1 , post1, "naive_em")
print(rse) 
    #mixk, postk, ll = kmeans.run(X, mix, post)
    #common.plot(X, mixk , postk, "kmeans")
  

    

   


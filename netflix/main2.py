import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
for i in range(1):
    mix, post   = common.init(X, 4,i)
    
    mix1, post1, ll = naive_em.run(X, mix, post)
    #common.plot(X, mix1 , post1, "naive_em")
    
    #mixk, postk, ll = kmeans.run(X, mix, post)
    #common.plot(X, mixk , postk, "kmeans")
  
    bic  = common.bic(X,mix1,  ll)
    print(bic)
    

   


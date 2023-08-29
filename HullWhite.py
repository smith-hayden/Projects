import pandas as pd
import numpy as np

rVar=np.random.normal(loc=0.0,scale=1.0,size=None)

def wProcess(price,t,i,vol):
    price0=(price*(1+i)**t)+(price*rVar*np.sqrt(t))

price=100
t=1
i=.055
vol=0.1

wProcess(price,t,i,vol)

import scipy as sp
from sp.optimize import minimize

term_struct=np.array([5.1,5.125,5.001,4.99,5.0,5.1,5.125,5.001,7,5.0,5.1,5.125,6.6,4.99,5.0,5.1,5.125,7.125,4.99,5.0,5.1,5.125,5.001,4.99,5.0,5.1,5.125,5.001,4.99,5.0]) # actual observations 
sim_term_struct=np.array([5.456,5.432,5.123,4.987,5.001,5.456,5.432,5.123,4.987,5.001,5.456,5.432,5.123,4.987,5.001,5.456,5.432,5.123,4.987,5.001,5.1,5.125,5.001,4.99,5.0,5.1,5.125,5.001,4.99,5.0]) # from Hull-White 

def rMse(params):
    theta,sigma=params
    #print(theta)
    print(sigma)
    predicted=theta+term_struct*sigma
    
    print(len(predicted))
    error=sim_term_struct-predicted
    return np.sqrt(np.mean(error**2))

initial_params=[5.5,0.05]
result=minimize(rMse,initial_params,method='Powell')

optimized_theta, optimized_sigma=result.x

# Print results
print(f"Optimized Theta: {optimized_theta}")
print(f"Optimized Sigma: {optimized_sigma}")
print(f"Optimized RMSE: {result.fun}")
print(result)

import pandas as pd
import numpy as np

def rVar():
    rand=np.random.normal(loc=0.0,scale=1.0,size=None)
    return rand

def wProcess(price,t,i,vol):
    price0=(price*(1+i)**t)+(price*rVar()*vol*np.sqrt(t))
    noterate=price0/price*100
    return noterate

price=100
t=1
i=.055
vol=0.05

wProcess(price,t,i,vol)


    
import scipy as sp
from sp.optimize import minimize

# observed interest rates
term_struct=np.array([5.1,5.125,5.001,4.99,5.0,5.1,5.125,5.001,7,5.0,5.1,5.125,6.6,4.99,5.0,5.1,5.125,7.125,4.99,5.0,5.1,5.125,5.001,4.99,5.0,5.1,5.125,5.001,4.99,5.0]) # actual observations 

# simulated interest rates from weiner process
sim_term_struct=np.array([5.456,5.432,5.123,4.987,5.001,5.456,5.432,5.123,4.987,5.001,5.456,5.432,5.123,4.987,5.001,5.456,5.432,5.123,4.987,5.001,5.1,5.125,5.001,4.99,5.0,5.1,5.125,5.001,4.99,5.0]) # from Hull-White 

# builds interest rate simulation array. Need to work on this
def sim_data():
    new_list=[]
    for n in term_struct:
        new_term_struct=(wProcess(price,t,n/100,vol))
        new_list.append(1-(new_term_struct-price)/n)

    return new_list
    
#print(np.mean(new_list)) # returns mean of forecasted interest rate change

def rMse(params,):
    theta,sigma=params
    predicted=theta+term_struct*sigma
    error=sim_term_struct-predicted
    return np.sqrt(np.mean(error**2))

initial_params=[5.0,0.05] # guess close to data

result=minimize(rMse,(initial_params),method='Powell')

optimized_theta, optimized_sigma=result.x # returns optimized theta, which is our optimized interest rate at time t
                                            #returns optimized sigma, which is our optimized volatility 

# Print results
print(np.mean(new_list)) # returns average forecasted change in i
print(f"Optimized Theta: {optimized_theta}") # mimized 
print(f"Optimized Sigma: {optimized_sigma}")
print(f"Optimized RMSE: {result.fun}")
#print(result)

# can this be discounted to present value and produce target interest rate given observations? Would this be an arbitrage opportunity?

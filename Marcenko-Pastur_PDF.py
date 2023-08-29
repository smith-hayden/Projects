# this proect is used for measuring the signal to noise ratio in financial datasets. As arbitrage decreases, so does data noise.

import numpy as np

import pandas as pd

def mpPDF(var,q,pts):
    # Marcenko-Pastur PDF
    # q=T/N LxW of matrix
    # var= variance 
    # pts= interval between data points for scale

    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts)
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf=pd.Series(pdf,index=eVal)
    return pdf


import sklearn
from sklearn.neighbors._kde import KernelDensity

def getPCA(matrix):
    # Get eigenvalues, eigenvectors, from Hermitian matrix
    eVal,eVec=np.linalg.eigh(matrix) # eVal is an array, eVec is a matrix
    indicies=eVal.argsort()[::1] # Arguments for sorting eigenvalues desc. Indicies is the position from where the original value was moved to the new value in the index.
                                # in this project, np.linalg.eigh retuend an ordered array assigned to eVal, which is why indicies is in ascending numeric order.
    
    eVal,eVec=eVal[indicies],eVec[:,indicies] # sets arrays eVal and eVec to the separate eVals and eVecs created by the .eigh() operator
   
    eVal=np.diagflat(eVal) # eigenvalues are stored in a matrix. .diagflat() takes the array and sets it to the diagonal of an empty matrix.
   
    return eVal, eVec

def fitKDE(obs, bWidth=.25,kernel='gaussian',x=None):
    # Fit kernel to a series of obs, and derive the probability of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None: x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1: x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf

x=np.random.normal(size=(10000,1000))
eVal0,eVec0=getPCA(np.corrcoef(x,rowvar=0))
pdf0=mpPDF(1.,q=x.shape[0]/float(x.shape[1]),pts=1000)
pdf1=fitKDE(np.diag(eVal0),bWidth=.01) # empirical pdf

# Adding signal to a random covariance matrix

def getRndCov(nCols,nFacts):
    w=np.random.normal(size=(nCols,nFacts)) #creates nCols x nFacts matrix of normally randomly distributed values
    cov=np.dot(w,w.T) # random cov matrix, however not full rank. calculates the dot product of w and the transpose of w. This creates a square covariance matrix.
    cov+=np.diag(np.random.uniform(size=nCols)) # full rank cov. Uniform randomly distributed values have equal distribution - adding this diagonal introduces a uncorrelated level of noise to the--
                                                # --random covariance matrix
    return cov # builds random covariance matrix

def cov2corr(cov):
    # derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(cov)) # because the covariance matrix is symmetrical, the diagonal contains the variance
    corr=cov/np.outer(std,std) # builds matrix of standard deviations then divides the covariance matrix by it to build correlation matrix
    corr[corr<-1],corr[corr>1]=-1,1 # numerical error. Where corr<-1, set the value to -1 and where corr>1, set the value to 1
    return corr

alpha,nCols,nFact,q=.995,1000,100,10 # q is a constant
cov=np.cov(np.random.normal(size=(nCols*q,nCols)),rowvar=0) # creates nCols*q x nCols matrix of normal randomly distributed variables. Rowvar is a bool so there are nCols columns in the matrix
cov=alpha*cov+(1-alpha)*getRndCov(nCols,nFact) # noise + signal. Sum of alpha-weighted random covariance matrix + 1-alpha * data matrix
corr0=cov2corr(cov)
eVal0,eVec0=getPCA(corr0)

# returns a matrix of eigenvalues and an array eigenvectors of the signal matrix

import scipy 
from scipy.optimize import minimize

def errPDFs(var,eVal,q,bWidth,pts=1000):
    # fit error. Produces sum of squared error
    pdf0=mpPDF(var[0],q,pts) #theoretical pdf
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values) #empirical pdf
    sse=np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal,q,bWidth):
    # find max random eVal by fitting Marcenko's dist
    out=minimize(lambda*x:errPDFs(*x),.5,args=(eVal,q,bWidth),bounds=((1E-5,1-1E-5),))
    if out['success']:var=out['x'][0]
    else: var=1
    eMax=var*(1+(1./q)**.5)**2
    return eMax,var

eMax0,var0=findMaxEval(np.diag(eVal0),q,bWidth=.01) # location of max eigenvalue
nFacts0=eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0) 


print(var0) # where on the y axis eMax0 value lands (probability) var0 - 1 = % of variance attributed to signal
print(nFacts0) # nFacts SHOULD equal the facts input in row 65

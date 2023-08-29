# Public Securities Association prepay modelling with SMM

import numpy as np
import pandas as pd

basePrepay=0.06 # minimum prepay assumption
prepayInc=0.002 # prepay increase per month
maxPrepay=20 # maximum prepay in %
months=360 # for a 30y conventional fixed
cprValues=[]

def psa_cpr(psa_factor,t):
    if t<31:
        cpr=psa_factor * (basePrepay + prepayInc * (t-1))
    else:
        cpr=maxPrepay
    return cpr

for t in range(1,months+1):
    cpr=psa_cpr(maxPrepay,t)
    cprValues.append(cpr)

for t, cpr in enumerate(cprValues[:35],start=1):

    print(f"Month {t} PSA CPR= {cpr:.4f}")

# Single Month Mortality (SMM) calc

SMM=[]
UPB=100000

for cpr in cprValues:
    mthlyCPR=cpr/12 # sets annual prepay to monthly
    prePmt=UPB*mthlyCPR # calcs prepay amount in $
    UPB-=prePmt # updates UPB to UPB after prepayment
    smm1=prePmt/UPB if UPB > 0 else 0 # if UPB is > 0, write the prePmt % to smm1, else set it to 0
    SMM.append(smm1)
    
print(SMM)
# work on why CPRs are negative after month 30

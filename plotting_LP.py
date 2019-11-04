"""
Created on Fri Feb  1 13:37:47 2019

@author: roger
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
for i in [0,1,2,3,4]:
    n=31
    drug_name = '_5fu'
    best = np.load('bestcost.npy')
    cost = np.loadtxt('param_'+str(i)+'_likelihoodprofile'+drug_name+'.txt')
    paramRange = np.loadtxt('param_'+str(i)+'_profilerange'+drug_name+'.txt')
    chi = chi2.isf(q=0.05, df=5)
    plt.figure()
    plt.plot(paramRange,cost)
    plt.plot(paramRange,[chi]*n)
    plt.ylim([0,chi*1.5])
    plt.show()
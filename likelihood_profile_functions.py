"""
Created on Tue Jan 29 14:12:38 2019

@author: roger
"""
from scipy.stats import chi2
import combination_functions_lohp as lohp
import combination_functions_cpt as cpt
import combination_functions_5fu as fu
import drug_data as pk
import cma
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools



def paramLikelihoodProfile(i=0,j=0,drug_type=0,n=50):
    """ i is parameter number,
    j is patient number,
    drug_type is 0,1,2,
    n is number of samples taken for th LP"""
    if drug_type == 0:
        pat_range = pk.pat_nums_cpt
        pat_data = [None]*2
        tot_dose = pk.CPT11_tot_dose[pat_range[j]]
        pat_data[0] = pk.CPT11[pat_range[j]]
        pat_data[1] = pk.SN38[pat_range[j]]
        data_time = pk.CPT11_time[pat_range[j]]
        drug_name = '_cpt11'
        cost_function = cpt.non_phys_param_cost_single
        args = (tot_dose,data_time,pat_data,2)
        k0 = np.loadtxt('data/pat_param_drug'+drug_name+'.txt')[j]
        best = cost_function(k0,tot_dose,data_time,pat_data,2)
        np.save('bestcost'+drug_name,best)
    elif drug_type == 1:
        pat_range =pk.pat_nums_fu
        tot_dose = pk.FU_tot_dose[pat_range[j]]
        pat_data = pk.FU[pat_range[j]]
        data_time = pk.FU_time[pat_range[j]]
        cost_function = fu.non_phys_param_cost_single
        drug_name = '_5fu'
        args = (tot_dose,data_time,pat_data)
        k0 = np.loadtxt('data/pat_param_drug'+drug_name+'.txt')[j]
        best = cost_function(k0,tot_dose,data_time,pat_data)
        np.save('bestcost'+drug_name,best)
    elif drug_type == 2:
        pat_range = pk.pat_nums_lohp
        pat_data = [None]*2
        tot_dose = pk.LOHP_tot_dose[pat_range[j]]
        pat_data[0] = pk.LOHP_free[pat_range[j]]
        pat_data[1] = pk.LOHP_total[pat_range[j]]
        data_time = pk.LOHP_time[pat_range[j]]
        cost_function = lohp.non_phys_param_cost_single_lohp
        drug_name = '_lohp'
        args = (tot_dose,data_time,pat_data)
        k0 = np.loadtxt('data/pat_param_drug'+drug_name+'.txt')[j]
        best = cost_function(k0,tot_dose,data_time,pat_data)
        np.save('bestcost'+drug_name,best)
    # number of repeats such that we are confident that the answer has not come from a local minimum
    rep = 10

    bounds = [0, 60000]
    cost = [0]*(n+1)
    paramRange = np.sort(np.insert(np.linspace(k0[i]*0.5,k0[i]*3,n),[0],k0[i]))
    for l in range(n+1):
        kiNew = paramRange[l]
        costRep = [None]*rep
        k1 = k0
        for k in range(rep):
            es = cma.CMAEvolutionStrategy(k1,10000,{'bounds': bounds,'verb_disp':0,'tolfun':1e-03,'fixed_variables':{i:kiNew}})
            es.optimize(cost_function,args=args)
            k1 = es.result[0]
            costRep[k] = es.result[1]
        cost[l] = costRep[costRep.index(min(costRep))]
    np.savetxt('param_'+str(i)+'patient_'+str(j)+'_likelihoodprofile'+drug_name+'.txt',cost)
    np.savetxt('param_'+str(i)+'patient_'+str(j)+'_profilerange'+drug_name+'.txt',paramRange)
    
def plot_LP(i=0,n=1,drug_type=0):
    if drug_type == 0:
        drug_name = '_cpt11'
        data_time = pk.CPT11_time[0]
    elif drug_type == 1:
        drug_name = '5fu'
        data_time = pk.FU_time[0]
    elif drug_type == 2:
        drug_name = '_lohp'
        data_time = pk.LOHP_time[0]
    best = np.load('bestcost'+drug_name+'.npy')
    cost = np.loadtxt('param_'+str(i)+'_likelihoodprofile'+drug_name+'.txt')
    paramRange = np.loadtxt('param_'+str(i)+'_profilerange'+drug_name+'.txt')
    chi = chi2.isf(q=0.05, df=len(data_time)) + best
    plt.figure()
    plt.plot(paramRange,cost)
    plt.plot(paramRange,[chi]*(n+1))
    plt.ylim([0,chi*1.1])
    plt.show()
    
if __name__ == '__main__':
#    iterables = [ [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9,10], [0] ]
#    input_array_cpt = [None]*110
#    for i,t in enumerate(itertools.product(*iterables)):
#        input_array_cpt[i] = (t)
#    iterables = [ [0,1,2,3,4], [0,1,2,3,4,5,6,7,8,9], [1] ]
#    input_array_5fu = [None]*50
#    for i,t in enumerate(itertools.product(*iterables)):
#        input_array_5fu[i] = (t)
#    iterables = [ [0,1,2,3,4,5,6], [0,1,2,3,4,5,6,7,8,9], [2] ]
#    input_array_lohp = [None]*70
#    for i,t in enumerate(itertools.product(*iterables)):
#        input_array_lohp[i] = (t)
    
        
#    input_array = input_array_cpt + input_array_5fu + input_array_lohp
#    i = input_array[int(sys.argv[1])][0]
#    j = input_array[int(sys.argv[1])][1]
#    drug_type = input_array[int(sys.argv[1])][2]
#    print(i,j,drug_type)
    i = 0
    j = 0
    drug_type=1
    paramLikelihoodProfile(i,j,drug_type)

        
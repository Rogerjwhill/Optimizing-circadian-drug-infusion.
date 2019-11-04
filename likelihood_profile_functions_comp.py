"""
Created on Tue Jan 29 14:12:38 2019

@author: roger
"""
import combination_functions_lohp as lohp
import combination_functions_cpt as cpt
import combination_functions_5fu as fu
import Useful_functions as uf
import drug_data as pk
import multiprocessing
import cma
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import chi2
import datetime as dt




def paramLikelihoodProfile(i=0,j=0,drug_type=0,n=20):
    """ i is parameter number,
    j is patient number,
    drug_type is 0,1,2,
    n is number of samples taken for th LP"""
    print('parameter=',i,'patient=',j,'drug type =',drug_type)
    if drug_type == 0:
        reps = 1
        cost_reps =[None]*reps
        pat_data = [None]*2
        tot_dose = pk.CPT11_tot_dose[0]
        pat_data[0] = pk.CPT11[0]
        pat_data[1] = pk.SN38[0]
        data_time = pk.CPT11_time[0]
        drug_name = '_cpt11'
        cost_function = cpt.non_phys_param_cost_single_both
        vol = pk.vol[0]
        args = (tot_dose,data_time,pat_data,vol)
        k0 = np.loadtxt('data/pat_param_drug'+drug_name+'.txt')[0]
        best = cost_function(k0,tot_dose,data_time,pat_data,vol)
        np.save('bestcost'+drug_name,best)
        bounds = [0, 1e+8]
        paramRange = [np.linspace(1.2e3,6e+3,n),np.linspace(0, 0.8e+4,n),np.linspace(3.5e4, 1.4e+5,n),np.linspace(1.6e4, 7e+4,n),np.linspace(0.7e3, 3e+3,n),np.linspace(3.5e4, 1.2e+5,n)]
    elif drug_type == 1:
        reps = 3
        cost_reps =[None]*reps
        tot_dose = pk.FU_tot_dose[0]
        pat_data = pk.FU[0]
        data_time = pk.FU_time[0]
        cost_function = fu.non_phys_param_cost_single
        drug_name = '_5fu'
        vol = pk.vol[0]
        args = (tot_dose,data_time,pat_data,vol)
        k0 = np.loadtxt('data/pat_param_drug'+drug_name+'.txt')[0]
        best = cost_function(k0,tot_dose,data_time,pat_data,vol)
        np.save('bestcost'+drug_name,best)
        bounds = [0, 1e+8]
        paramRange = [np.linspace(1.2e4,6e5,n),np.linspace(0,1.8e5,n),np.linspace(2,1.2e2,n)]
    elif drug_type == 2:
        reps = 1
        cost_reps =[None]*reps
        pat_data = [None]*2
        tot_dose = pk.LOHP_tot_dose[0]
        pat_data[0] = pk.LOHP_free[0]
        pat_data[1] = pk.LOHP_total[0]
        data_time = pk.LOHP_time[0]
        cost_function = lohp.non_phys_param_cost_single_lohp
        drug_name = '_lohp'
        vol = pk.vol[0]
        args = (tot_dose,data_time,pat_data,vol)
        k0 = np.loadtxt('data/pat_param_drug'+drug_name+'.txt')[0]
        best = cost_function(k0,tot_dose,data_time,pat_data,vol)
        np.save('bestcost'+drug_name,best)
        bounds = [0, 1e+8]
        paramRange = [np.linspace(0.9e4,3e4,n),np.linspace(1.1e4,5.1e04,n),np.linspace(4e03,2e04,n),np.linspace(8e03,1.6e04,n),np.linspace(3e02,7e2,n)]


    cost = [0]*(n+1)
    pRange = np.sort(np.append(paramRange[i],k0[i]))
    for l in range(n+1):
        kiNew = pRange[l]
        k1 = k0
        for rep in range(reps):
            es = cma.CMAEvolutionStrategy(k1,1e04,{'bounds': bounds,'verb_log': 0,'verb_disp': 0,'tolfun':1e-02,'fixed_variables':{i:kiNew}})
            es.optimize(cost_function,args=args)
#            k1 = es.result[0]
            cost_reps[rep] = es.result[1]
        cost[l] = np.min(cost_reps)
    np.savetxt('param_'+str(i)+'patient_'+str(j)+'_likelihoodprofile'+drug_name+'.txt',cost)
    np.savetxt('param_'+str(i)+'patient_'+str(j)+'_profilerange'+drug_name+'.txt',pRange)
    return cost, pRange, k0[i], best

def function_call_cpt(i):
    drug_type = 0
    j = 0
    cost, pRange, k, best = paramLikelihoodProfile(i,j,drug_type)
    np.savez('cost_pRange_k_best_cpt_param_'+str(i)+'.npz',cost=cost,pRange=pRange,k=k,best=best)
    return

def function_call_fu(i):
    drug_type = 1
    j = 0
    cost, pRange, k, best = paramLikelihoodProfile(i,j,drug_type)
    np.savez('cost_pRange_k_best_fu_param_'+str(i)+'.npz',cost=cost,pRange=pRange,k=k,best=best)
    return

def function_call_lohp(i):
    drug_type = 2
    j = 0
    cost, pRange, k, best = paramLikelihoodProfile(i,j,drug_type)
    np.savez('cost_pRange_k_best_lohp_param_'+str(i)+'.npz',cost=cost,pRange=pRange,k=k,best=best)
    return



def plot_LL_cpt(i=0):
    labels = ['Clearance$_{cpt,O/L}$','Clearance$_{cpt,B}$','Clearance$_{sn,O/L}$','Bioactivation$_{cpt}$','transport Blood-Liver','transport Blood-Organ']
    data = np.load('cost_pRange_k_best_cpt_param_'+str(i)+'.npz')
    cost = data['cost']
    pRange = data['pRange']
    k = data['k']
    best = data['best']
    Chi = chi2.isf(q=0.05, df=1)
    fig = plt.figure()
    plt.plot(pRange,cost,label='Likelihood profile')
    chi = Chi + best
    plt.plot(pRange,chi.repeat(len(pRange)),label='Confidence threshold')
    plt.plot(k,best,'rx',label='Minimum')
    plt.ylim([best*0.99,chi*1.1])
    plt.title('CPT11 '+labels[i])
    if i == 3:
        plt.xlabel('param value (mg/h)')
    else:
        plt.xlabel('param value (ml/h)')
    plt.ylabel('score')
    plt.legend()
    plt.show()
    with PdfPages('likelihood_profile_cpt_parameter_'+str(i)+'.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')

def plot_LL_fu(i=0):
    labels =['Clearance$_{O/L}$','Clearance$_B$','Uptake/Efflux$_{O/L}$']
    data = np.load('cost_pRange_k_best_fu_param_'+str(i)+'.npz')
    cost = data['cost']
    pRange = data['pRange']
    k = data['k']
    best = data['best']
    Chi = chi2.isf(q=0.06, df=1) 
    fig = plt.figure()
    plt.plot(pRange,cost,label='Likelihood profile')
    chi = Chi + best
    plt.plot(pRange,chi.repeat(len(pRange)),label='Confidence threshold')
    plt.plot(k,best,'rx',label='Minimum')
    plt.ylim([best*0.99,chi*1.1])
    plt.title('5-FU '+labels[i])
    plt.xlabel('param value (ml/h)')
    plt.ylabel('score')
    plt.legend()
    plt.show()
    with PdfPages('likelihood_profile_fu_parameter_'+str(i)+'.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        
def plot_LL_lohp(i=0):
    labels = ['Clearance$_{O/L}$','Clearance$_B$','Efflux/Uptake$_L$','Efflux/Uptake$_O$','Bind','Unbind']
    data = np.load('cost_pRange_k_best_lohp_param_'+str(i)+'.npz')
    cost = data['cost']
    pRange = data['pRange']
    k = data['k']
    best = data['best']
    Chi = chi2.isf(q=0.05, df=1) 
    fig = plt.figure()
    plt.plot(pRange,cost,label='Likelihood profile')
    chi = Chi + best
    plt.plot(pRange,chi.repeat(len(pRange)),label='Confidence threshold')
    plt.plot(k,best,'rx',label='Minimum')
    plt.ylim([best*0.99,chi*1.1])
    plt.title('LOHP '+labels[i])
    plt.xlabel('param value (ml/h)')
    plt.ylabel('score')
    plt.legend()
    plt.show()
    with PdfPages('likelihood_profile_lohp_parameter_'+str(i)+'.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')


    
    
if __name__ == '__main__':
    
    # cpt = 0 4h for 5 samples, fu = 1 5mins for 5 samples, lohp = 1 48mins for 3 samples.
    for drug in [0,2]:
    
        input_array = [[6,0],[3,1],[5,2]]
        i = input_array[drug][0]
        j = 0
        drug_type = input_array[drug][1]
        run = 1
    
        param_range = [x for x in range(i)]
#        param_range = [x for x in [1,0]]
        if drug == 0:
            print('cpt version 2')
            print('Start time: ',dt.datetime.now().time())
            # takes about 10 mins for 10 repeats 
            # paralel processing
            if run ==1:
                uf.tic()
                p = multiprocessing.Pool(i)
                p.map(function_call_cpt, param_range)
                p.close()
                p.join()
                uf.toc()
                
            for i in range(i):
                plot_LL_cpt(i)
                
        elif drug == 1:
            print('fu version 2')
            print('Start time: ',dt.datetime.now().time())
            # takes about 10 mins for 10 repeats 
            # paralel processing
            if run ==1:
                uf.tic()
                p = multiprocessing.Pool(i)
                p.map(function_call_fu, param_range)
                p.close()
                p.join()
                uf.toc()
            
            for i in range(i):
                plot_LL_fu(i)
                
        elif drug == 2:
            print('lohp version 2')
            print('Start time: ',dt.datetime.now().time())
            # takes about 10 mins for 10 repeats 
            # paralel processing
            if run ==1:
                uf.tic()
                p = multiprocessing.Pool(i)
                p.map(function_call_lohp, param_range)
                p.close()
                p.join()
                uf.toc()
            
            for i in range(i):
                plot_LL_lohp(i)
        

    
#    uf.tic()
#    function_call_cpt(0)
#    function_call_fu(0)
#    function_call_lohp(0)
#    uf.toc()
#    plot_LL_fu(0)

        

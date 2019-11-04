"""
Created on Fri Oct  6 14:23:52 2017

@author: roger
"""

import combination_functions_lohp as lohp
import combination_functions_cpt as cpt
import combination_functions_5fu as fu
import drug_data as pk
import Useful_functions as uf
import cma
import numpy as np
import multiprocessing
from functools import partial
import datetime as dt
import os 

def cmaes_run_pooled(drug_type = 2,repeats = 10,save = True):
    if drug_type == 0:
        drug_name = '_cpt11'
        cost_function = cpt.pooled_data_cost
        x0 = [15000]*6
        bounds = [0, 2e+8]
    elif drug_type == 1:
        cost_function = fu.pooled_data_cost
        drug_name = '_5fu'
        x0 = [15000]*3
        bounds = [0, 1e+7]
    elif drug_type == 2:
        cost_function = lohp.pooled_data_cost
        drug_name = '_lohp'
        x0 = [15000]*6
        bounds = [0, 1e+8]


    res = [None]*repeats
    cost = [None]*repeats
    for i in range(repeats):
        es = cma.CMAEvolutionStrategy(x0,1e03,{'bounds': bounds,'verb_disp':0,'verb_log':0,'tolfun':1e-1})
        es.optimize(cost_function)
        res[i] = es.result[0]
        cost[i] = es.result[1]
        x0 = es.result[0]
        print(drug_name,'repeats',i+1,'out of', repeats)
    best = cost.index(min(cost))
    m_res = res[best]


    if save:
        np.savetxt('data/pooled_param_drug'+drug_name+'.txt',m_res)
        if repeats > 1:
            np.save('data/pooled_all_repeats_'+str(drug_type)+'_'+str(repeats),res)
            np.save('data/pooled_all_costs_'+str(drug_type)+'_'+str(repeats),cost)
    return m_res


def cmaes_run_non_phys_single(drug_type = 2,repeats = 3,save = True):
    if drug_type == 0:
        pat_range = pk.pat_nums_cpt
    elif drug_type == 1:
        pat_range =pk.pat_nums_fu
    elif drug_type == 2:
        pat_range = pk.pat_nums_lohp
    m_res = [None]*len(pat_range)
    T_res = [None]*len(pat_range)
    cost = [None]*len(pat_range)
    for j in range(0,len(pat_range)):
        if drug_type == 0:
            pat_data = [None]*2
            tot_dose = pk.CPT11_tot_dose[pat_range[j]]
            pat_data[0] = pk.CPT11[pat_range[j]]
            pat_data[1] = pk.SN38[pat_range[j]]
            vol = pk.vol[pat_range[j]]
            data_time = pk.CPT11_time[pat_range[j]]
            drug_name = '_cpt11'
            cost_function = cpt.non_phys_param_cost_single_both
            plotter = cpt.non_phys_plotter_data_model
            x0 = [15000]*6
            bounds = [0, 2e+8]
            args = (tot_dose,data_time,pat_data,vol)
        elif drug_type == 1:
            tot_dose = pk.FU_tot_dose[pat_range[j]]
            pat_data = pk.FU[pat_range[j]]
            data_time = pk.FU_time[pat_range[j]]
            vol = pk.vol[pat_range[j]]
            cost_function = fu.non_phys_param_cost_single
            plotter = fu.non_phys_plotter_data_model
            drug_name = '_5fu'
            x0 = [15000]*3
            bounds = [0, 1e+7]
            args = (tot_dose,data_time,pat_data,vol)
        elif drug_type == 2:
            pat_data = [None]*2
            tot_dose = pk.LOHP_tot_dose[pat_range[j]]
            pat_data[0] = pk.LOHP_free[pat_range[j]]
            pat_data[1] = pk.LOHP_total[pat_range[j]]
            vol = pk.vol[pat_range[j]]
            data_time = pk.LOHP_time[pat_range[j]]
            cost_function = lohp.non_phys_param_cost_single_lohp
            plotter = lohp.non_phys_plotter_data_model
            drug_name = '_lohp'
            x0 = [15000]*6
            bounds = [0, 1e+8]
            args = (tot_dose,data_time,pat_data,vol)


        res = [None]*repeats
        cost[j] = [None]*repeats
        for i in range(repeats):
            es = cma.CMAEvolutionStrategy(x0,1e03,{'bounds': bounds,'verb_disp':0,'verb_log':0,'tolfun':1e-1})
            es.optimize(cost_function,args=args)
            res[i] = es.result[0]
            cost[j][i] = es.result[1]
            x0 = es.result[0]
            print(drug_name,'patient', j+1,'of',len(pat_range),'repeats',i+1,'out of', repeats)
        best = cost[j].index(min(cost[j]))
        m_res[j] = res[best]
        T_res[j] = res

    if save:
        np.savetxt('data/pat_param_drug'+drug_name+'.txt',m_res)
        if repeats > 1:
            np.save('data/all_repeats_'+str(drug_type)+'_'+str(repeats),T_res)
            np.save('data/all_costs_'+str(drug_type)+'_'+str(repeats),cost)
        plotter(m_res)
    return m_res

def cmaes_run_non_phys_parallel(drug_type = 2):
    if drug_type == 0:
        drug_name = '_cpt11'
        pat_range = pk.pat_nums_cpt
    elif drug_type == 1:
        drug_name = '_5fu'
        pat_range =pk.pat_nums_fu
    elif drug_type == 2:
        drug_name = '_lohp'
        pat_range = pk.pat_nums_lohp
    n = len(pat_range)
    # paralel processing
    p = multiprocessing.Pool()
    fun = partial(cmaes_run_par,drug_type=drug_type)
    out = p.map(fun, range(n))
    p.close()
    p.join()
    param = [None]*n
    res = [None]*n
    cost = [None]*n
    for x in range(n):
        param[x] = out[x][1]
        res[x] = out[x][2]
        cost[x] = out[x][3]
    if os.path.isfile('data/pat_param_drug'+drug_name+'.txt'):
        os.remove('data/pat_param_drug'+drug_name+'.txt')
    if os.path.isfile('data/all_repeats_'+str(drug_type)+'.npy'):
        os.remove('data/all_repeats_'+str(drug_type)+'.npy')
    if os.path.isfile('data/all_costs_'+str(drug_type)+'.npy'):
        os.remove('data/all_costs_'+str(drug_type)+'.npy')
    np.savetxt('data/pat_param_drug'+drug_name+'.txt',param)
    np.save('data/all_repeats_'+str(drug_type)+'.npy',res)
    np.save('data/all_costs_'+str(drug_type)+'.npy',cost)
    return out
    
def cmaes_run_par(j=0,drug_type = 0):
    if drug_type == 0:
        pat_range = pk.pat_nums_cpt
        pat_data = [None]*2
        tot_dose = pk.CPT11_tot_dose[pat_range[j]]
        pat_data[0] = pk.CPT11[pat_range[j]]
        pat_data[1] = pk.SN38[pat_range[j]]
        data_time = pk.CPT11_time[pat_range[j]]
        cost_function = cpt.non_phys_param_cost_single_both
        x0 = [10000]*6
        bounds = [0, 2e+8]
        sigma = 1e04
        vol = pk.vol[pat_range[j]]
        args = (tot_dose,data_time,pat_data,vol)
    elif drug_type == 1:
        pat_range =pk.pat_nums_fu
        tot_dose = pk.FU_tot_dose[pat_range[j]]
        pat_data = pk.FU[pat_range[j]]
        data_time = pk.FU_time[pat_range[j]]
        cost_function = fu.non_phys_param_cost_single
        x0 = [15000]*3
        bounds = [0, 1e+7]
        sigma = 1e04
        vol = pk.vol[pat_range[j]]
        args = (tot_dose,data_time,pat_data,vol)
    elif drug_type == 2:
        pat_range = pk.pat_nums_lohp
        pat_data = [None]*2
        tot_dose = pk.LOHP_tot_dose[pat_range[j]]
        pat_data[0] = pk.LOHP_free[pat_range[j]]
        pat_data[1] = pk.LOHP_total[pat_range[j]]
        data_time = pk.LOHP_time[pat_range[j]]
        cost_function = lohp.non_phys_param_cost_single_lohp
        x0 = [15000]*5
        bounds = [0, 1e+8]
        sigma = 1e04
        vol = pk.vol[pat_range[j]]
        args = (tot_dose,data_time,pat_data,vol)
        

    repeats = 4
    res = [None]*repeats
    cost = [None]*repeats
    for i in range(repeats):
        print('patient', j+1,'of',len(pat_range),'repeats',i,'out of', repeats)
        es = cma.CMAEvolutionStrategy(x0,sigma,{'bounds': bounds,'verb_disp':0,'verb_log':0,'tolfun':1e-2})
        es.optimize(cost_function,args=args)
        res[i] = es.result[0]
        cost[i] = es.result[1]
        x0 = es.result[0]
    best_param = res[cost.index(min(cost))]
    return j , best_param , res, cost

if __name__ == '__main__':
    print('Start time: ',dt.datetime.now().time())
    

#    uf.tic()
#    # paralel processing
#    p = multiprocessing.Pool(3)
#    p.map(cmaes_run_non_phys_single, gen1)
#    p.close()
#    p.join()
#    uf.toc()

    for i in [0]:
        uf.tic()
        out = cmaes_run_non_phys_parallel(drug_type = i)
        uf.toc()

#    for i in [0]:
#        cmaes_run_pooled(i)





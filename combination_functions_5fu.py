
"""
Created on Fri Oct  6 14:11:57 2017

@author: roger
"""
import drug_delivery as dd
import drug_data as pk
import Useful_functions as uf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import odeint

def non_phys_ode_5_fu(x,t,k,tot_dose,vol):
    c_l = k[0]
    c_b = k[1]
    ef_l = k[2]*vol[0]
    up_l = k[2]*vol[0]
    ef_o = k[2]*vol[2]
    up_o = k[2]*vol[2]



    liver = x[0]
    Blood = x[1]
    organs = x[2]
    clear_b = x[3]
    clear_l = x[4]

    

    drug_input = dd.drug_del_5_fu


    dxdt = [ (drug_input(t,tot_dose) - c_l*liver - ef_l*liver + up_l*Blood)/vol[0],
            (ef_l*liver + ef_o*organs - up_o*Blood - up_l*Blood - c_b*Blood )/vol[1],
            (up_o*Blood - ef_o*organs)/vol[2],
            c_b*Blood,
            c_l*liver]
    return dxdt

def pooled_data_cost(k):
    pat_range = pk.pat_nums_fu
    cost = 0
    for j in range(0,len(pat_range)):
        pat_range =pk.pat_nums_fu
        tot_dose = pk.FU_tot_dose[pat_range[j]]
        pat_data = pk.FU[pat_range[j]]
        data_time = pk.FU_time[pat_range[j]]
        vol = pk.vol[pat_range[j]]
        cost += non_phys_param_cost_single(k,tot_dose,data_time,pat_data,vol)
    return cost

def non_phys_param_cost_single(k,pat_dose,data_time,pat_data,vol):
    cost = 0
    x0 = [0,0,0,0,0]
    tspan = data_time
    t = np.sort(list(np.linspace(20,35,10000))+list(tspan))
    t = np.sort(list(set(t)))
    pos = np.where(np.in1d(t,tspan))[0]
    x = odeint(non_phys_ode_5_fu,x0,t,(k,pat_dose,vol),hmax=1)
    M_u = x[pos,1]
    D_u = pat_data
    drug_clear = (x[-1,4]/(x[-1,4] + x[-1,3]) - 0.8)**2
    cost = np.nansum(np.square((M_u - D_u)/(3.71631239e-05))) + drug_clear*1.1e2
    return cost

def non_phys_param_cost_PL(k,ki,i,pat_dose,data_time,pat_data):
    k[i] = ki
    cost = 0
    x0 = [0,0,0]
    tspan = data_time
    t = np.sort(list(np.linspace(20,35,10000))+list(tspan))
    t = np.sort(list(set(t)))
    pos = np.where(np.in1d(t,tspan))[0]
    x = odeint(non_phys_ode_5_fu,x0,t,(k,pat_dose),hmax=1)
    M_u = x[pos,1]
    D_u = pat_data
    cost = np.nansum(np.square((M_u - D_u)/1.5e-05))
    return cost

def evaluate_pk_cost(values,tot_dose,data_time,pat_data):
    Y = np.zeros([values.shape[0]])
    for i, X in enumerate(values):
        Y[i] = non_phys_param_cost_single(X,tot_dose,data_time,pat_data)
    return Y

def non_phys_plotter_data_model(k=[],save = False):
    if k == []:
        k = np.loadtxt('data/pat_param_drug_5fu.txt')

    tot_dose = pk.FU_tot_dose
    pat_data = pk.FU
    pat_range = pk.pat_nums_fu
    vol = pk.vol
    data_time = pk.FU_time
    Title_str = '5-FU'


#    non_phys_param_cost_single(k[0],tot_dose[0],data_time[0],pat_data[0],vol[0])
    x0 = [0,0,0,0,0]
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,a,sharex = True, sharey = True)
#    fig0.suptitle(Title_str+' model vs data')
    cmax_fu = [None]*len(pat_range)
    cmax_time = [None]*len(pat_range)
    cost = [None]*len(pat_range)
    R2 = [None]*len(pat_range)
    auc_ratio = [None]*len(pat_range)
    auc = [None]*len(pat_range)
    
    for i in range(0,len(pat_range)):
        tspan = data_time[pat_range[i]]
        t = np.sort(list(np.linspace(20,38,10000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_5_fu,x0,t,(k[i],tot_dose[pat_range[i]],vol[pat_range[i]]),hmax=5)
        x = np.multiply(1000,x)
        spike_start = np.where(t>33.75)[0][0]
        auc_spike = np.trapz(x[spike_start:-1,1],t[spike_start:-1])
        auc_main = np.trapz(x[0:spike_start,1],t[0:spike_start])
        auc_ratio[i] = (auc_spike/auc_main)*100
        auc[i] = np.trapz(x[:,2],t)

        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,1],'k--')
        ax0[int(np.floor(i/a)),i%a].errorbar(tspan,np.multiply(1000,pat_data[pat_range[i]]),yerr=1.5e-2,fmt='.',color='0.4',mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(31,1.5,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=70 )
        cmax_fu[i] = np.max(x[:,1])
        cmax_time[i] = t[np.argmax(x[:,1])]
        M_u = x[pos,1]
        D_u = np.multiply(1000,pat_data[pat_range[i]])
        cost[i] = np.nansum(np.square((M_u - D_u)/1.5e-02))
        R2[i] = 1 - np.nansum(np.square(M_u - D_u))/np.nansum((D_u - np.nanmean(D_u))**2)


        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
    if a%2 != 0:
        fig0.delaxes(ax0[-1,-1])
    ax0[int(np.floor(i/a)),i%a].legend(['model','data'],bbox_to_anchor=(1,1))
    plt.setp(ax0,xlim=[22,37], xticks=np.arange(22,37,6),xticklabels=np.arange(22,37,6)%24,ylim=[0,np.max([np.max(cmax_fu),np.nanmax(pat_data)])*1.2])
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(0.05,0.65,'Concentration (mg/ml)',ha='right',rotation='vertical')
    if save:
        np.savetxt('cmax_fu_patients.txt',cmax_fu)
        np.savetxt('cmax_fu_time.txt',cmax_time)
        folder_path = uf.new_folder('Model_fitting')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'.pdf') as pdf:
            pdf.savefig(fig0, bbox_inches='tight')

    return x,cost, R2

def non_phys_plotter_data_model_colour(k=[],save = False):
    if k == []:
        k = np.loadtxt('data/pat_param_drug_5fu.txt')

    tot_dose = pk.FU_tot_dose
    pat_data = pk.FU
    pat_range = pk.pat_nums_fu
    data_time = pk.FU_time
    Title_str = '5-FU'


    
    x0 = [0,0,0]
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,a,sharex = True, sharey = True)
    fig0.suptitle('5-Fu model fit to data')
    cmax_fu = [None]*len(pat_range)
    cmax_time = [None]*len(pat_range)
    cost = [None]*len(pat_range)
    R2 = [None]*len(pat_range)
    
    for i in range(0,len(pat_range)):
        tspan = data_time[pat_range[i]]
        t = np.sort(list(np.linspace(20,38,10000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_5_fu,x0,t,(k[i],tot_dose[pat_range[i]]),hmax=5)
        

        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,1],'--',color=(0,0,0.5))
        ax0[int(np.floor(i/a)),i%a].plot(tspan,pat_data[pat_range[i]],marker='.',color=(0.7,0,0),mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(31,0.0015,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=70 )
        cmax_fu[i] = np.max(x[:,1])
        cmax_time[i] = t[np.argmax(x[:,1])]
        M_u = x[pos,1]
        D_u = pat_data[pat_range[i]]
        cost[i] = np.nansum(np.square(M_u - D_u))
        R2[i] = 1 - np.nansum(np.square((M_u - D_u)))/np.nansum((D_u - np.nanmean(D_u))**2)


        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
    if a%2 != 0:
        fig0.delaxes(ax0[-1,-1])
    ax0[int(np.floor(i/a)),i%a].legend(['model','data'],bbox_to_anchor=(1,1))
    plt.setp(ax0,xlim=[22,37], xticks=np.arange(22,37,6),xticklabels=np.arange(22,37,6)%24,ylim=[0,np.max([np.max(cmax_fu),np.nanmax(pat_data)])*1.1])
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(0,0.65,'Concentration (mg/ml)',ha='right',rotation='vertical')
    if save:
        np.savetxt('cmax_fu_patients.txt',cmax_fu)
        np.savetxt('cmax_fu_time.txt',cmax_time)
        folder_path = uf.new_folder('Model_fitting')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'_colour.pdf') as pdf:
            pdf.savefig(fig0, bbox_inches='tight')

    return x,cost,auc_ratio

def plot_pooled_data(k=None):
    if k == None:
        k = np.loadtxt('data/pooled_param_drug_5fu.txt')

    tot_dose = pk.FU_tot_dose
    pat_data = pk.FU
    pat_range = pk.pat_nums_fu
    data_time = pk.FU_time
    vol = pk.vol
    Title_str = '5-FU'


    
    x0 = [0,0,0,0,0]
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,a,sharex = True, sharey = True)
    fig0.suptitle('5-Fu model fit to data')
    cmax_fu = [None]*len(pat_range)
    cmax_time = [None]*len(pat_range)
    cost = [None]*len(pat_range)
    SS_tot = [None]*len(pat_range)
    SS_res = [None]*len(pat_range)
    
    for i in range(0,len(pat_range)):
        tspan = data_time[pat_range[i]]
        t = np.sort(list(np.linspace(20,38,10000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_5_fu,x0,t,(k,tot_dose[pat_range[i]],vol[pat_range[i]]),hmax=5)
        

        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,1],'--',color=(0,0,0.5))
        ax0[int(np.floor(i/a)),i%a].plot(tspan,pat_data[pat_range[i]],'.',color=(0.7,0,0),mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(31,0.0015,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=70 )
        cmax_fu[i] = np.max(x[:,1])
        cmax_time[i] = t[np.argmax(x[:,1])]
        M_u = x[pos,1]
        D_u = pat_data[pat_range[i]]
        cost[i] = np.nansum(np.square(M_u - D_u))
        SS_tot[i] = np.nansum((D_u - np.nanmean(D_u))**2)
        SS_res[i] = np.nansum(np.square((M_u - D_u)))
    Cost = np.sum(cost)
    r2 = 1 - np.sum(SS_res)/np.sum(SS_tot)

    plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
    if a%2 != 0:
        fig0.delaxes(ax0[-1,-1])
    ax0[int(np.floor(i/a)),i%a].legend(['model','data'],bbox_to_anchor=(1,1))
    plt.setp(ax0,xlim=[22,37], xticks=np.arange(22,37,6),xticklabels=np.arange(22,37,6)%24,ylim=[0,np.max([np.max(cmax_fu),np.nanmax(pat_data)])*1.1])
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(0,0.65,'Concentration (mg/ml)',ha='right',rotation='vertical')
    
    np.savetxt('cmax_fu_patients.txt',cmax_fu)
    np.savetxt('cmax_fu_time.txt',cmax_time)
    folder_path = uf.new_folder('Model_fitting')
    with PdfPages(folder_path+'model_fit_for_pooled'+Title_str+'_colour.pdf') as pdf:
        pdf.savefig(fig0, bbox_inches='tight')

    return Cost, r2


if __name__ == '__main__':
#    x,cost, R2 = non_phys_plotter_data_model(save=True)
#    r2 = np.round(np.transpose(R2),2)
#    b = np.round(cost,2)
#    print(b)
    cost, r2 = plot_pooled_data()
    print('5-fu pde SSR = ',cost)
    print('5-fu pde r2 = ',r2)

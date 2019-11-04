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

def non_phys_ode_CPT(x,t,k,tot_dose,vol):
    
    c_cpt_l = k[0]
    c_cpt_o = k[0]
    c_cpt_b = k[1]
    c_sn_l = k[2]
    c_sn_o = k[2]
    Vcp = k[3]
    ef_l = k[4]
    up_l = k[4]
    ef_o = k[5]
    up_o = k[5]
   


    liver_cp = x[0]
    liver_sn = x[1]
    Blood_cp = x[2]
    Blood_sn = x[3]
    organs_cp = x[4]
    organs_sn = x[5]
    clear_b_cpt = x[6]
    clear_l_cpt = x[7]
    clear_o_cpt = x[8]
    clear_l_sn = x[9]
    clear_o_sn = x[10]
    
    Kcp = 29 #ug/ml

    drug_input = dd.drug_del_CPT11


    dxdt = [ (drug_input(t,tot_dose) - c_cpt_l*liver_cp - ef_l*liver_cp + up_l*Blood_cp - (Vcp*liver_cp)/(Kcp+liver_cp))/vol[0],
            (- c_sn_l*liver_sn - ef_l*liver_sn + up_l*Blood_sn + 0.67*(Vcp*liver_cp)/(Kcp+liver_cp) )/vol[0],
            (ef_l*liver_cp - up_l*Blood_cp + ef_o*organs_cp - up_o*Blood_cp - c_cpt_b*Blood_cp )/vol[1],
            (ef_l*liver_sn - up_l*Blood_sn + ef_o*organs_sn - up_o*Blood_sn)/vol[1],
            (up_o*Blood_cp - ef_o*organs_cp - c_cpt_o*organs_cp - (Vcp*organs_cp)/(Kcp + organs_cp))/vol[2],
            (up_o*Blood_sn - ef_o*organs_sn - c_sn_o*organs_sn + 0.67*(Vcp*organs_cp)/(Kcp + organs_cp))/vol[2],
            c_cpt_b*Blood_cp,
            c_cpt_l*liver_cp,
            c_cpt_o*organs_cp,
            c_sn_l*liver_sn,
            c_sn_o*organs_sn]
    return dxdt
def non_phys_ode_CPT_systemic(x,t,k,tot_dose,vol):
    
    c_cpt_l = k[0]
    c_cpt_o = k[0]
    c_cpt_b = k[1]
    c_sn_l = k[2]
    c_sn_o = k[2]
    Vcp = k[3]
    ef_l = k[4]
    up_l = k[4]
    ef_o = k[5]
    up_o = k[5]
   


    liver_cp = x[0]
    liver_sn = x[1]
    Blood_cp = x[2]
    Blood_sn = x[3]
    organs_cp = x[4]
    organs_sn = x[5]
    clear_b_cpt = x[6]
    clear_l_cpt = x[7]
    clear_o_cpt = x[8]
    clear_l_sn = x[9]
    clear_o_sn = x[10]
    
    Kcp = 29 #ug/ml

    drug_input = dd.drug_del_CPT11


    dxdt = [ ( - c_cpt_l*liver_cp - ef_l*liver_cp + up_l*Blood_cp - (Vcp*liver_cp)/(Kcp+liver_cp))/vol[0],
            (- c_sn_l*liver_sn - ef_l*liver_sn + up_l*Blood_sn + 0.67*(Vcp*liver_cp)/(Kcp+liver_cp) )/vol[0],
            (ef_l*liver_cp - up_l*Blood_cp + ef_o*organs_cp - up_o*Blood_cp - c_cpt_b*Blood_cp )/vol[1],
            (ef_l*liver_sn - up_l*Blood_sn + ef_o*organs_sn - up_o*Blood_sn)/vol[1],
            (up_o*Blood_cp - ef_o*organs_cp - c_cpt_o*organs_cp - (Vcp*organs_cp)/(Kcp + organs_cp))/vol[2],
            (up_o*Blood_sn - ef_o*organs_sn - c_sn_o*organs_sn + 0.67*(Vcp*organs_cp)/(Kcp + organs_cp))/vol[2],
            c_cpt_b*Blood_cp,
            c_cpt_l*liver_cp,
            c_cpt_o*organs_cp,
            c_sn_l*liver_sn,
            c_sn_o*organs_sn]
    return dxdt

def pooled_data_cost(k):
    pat_range = pk.pat_nums_cpt
    cost = 0
    for j in range(0,len(pat_range)):
        pat_data = [None]*2
        tot_dose = pk.CPT11_tot_dose[pat_range[j]]
        pat_data[0] = pk.CPT11[pat_range[j]]
        pat_data[1] = pk.SN38[pat_range[j]]
        vol = pk.vol[pat_range[j]]
        data_time = pk.CPT11_time[pat_range[j]]
        cost += non_phys_param_cost_single_both(k,tot_dose,data_time,pat_data,vol)
    return cost

def non_phys_param_cost_single_both(k,pat_dose,data_time,pat_data,vol):

    cost = 0
    x0 = [0]*11
    tspan = data_time
    t = np.sort(list(np.linspace(0.1,24,1000))+list(tspan))
    t = np.sort(list(set(t)))
    pos = np.where(np.in1d(t,tspan))[0]
    x = odeint(non_phys_ode_CPT,x0,t,(k,pat_dose,vol),hmax=1)
    M_cp = x[pos,2]
    M_sn = x[pos,3]
    D_cp = pat_data[0]
    D_sn = pat_data[1]
#    x0[2] = pat_dose
    #systemic clearance conditions
#    x = odeint(non_phys_ode_CPT_systemic,x0,t,(k,pat_dose,vol),hmax=1)
    clear_tot = x[-1,6] + x[-1,7] + x[-1,8] + 1.72*x[-1,9] + 1.72*x[-1,10]
    total_cleared = 100*(clear_tot/pat_dose - 1)**2
    total_sn_cleared = 100*(1.72*(x[-1,9]+ x[-1,10])/pat_dose - 0.15)**2
    
#    clear_sn = 1*(x[-1,10]/x[-1,9] - 2)**2
    clear_b = 200*(x[-1,6]/clear_tot - 0.25)**2
#    clear_l = 1*(x[-1,7]/clear_tot - 0.30)**2
#    clear_o = 1*(x[-1,8]/clear_tot - 0.30)**2
    
    systemic_conditions = total_cleared + total_sn_cleared + clear_b#+ clear_sn +clear_b + clear_o + clear_l
    # total cost
    cost = np.nansum(np.square((M_cp - D_cp)/(2.19317759e-05)) + np.square((M_sn - D_sn)/(6.81021571e-07))) + systemic_conditions
    return cost

def evaluate_pk_cost(values,tot_dose,data_time,pat_data):
    Y = np.zeros([values.shape[0]])
    for i, X in enumerate(values):
        Y[i] = non_phys_param_cost_single_both(X,tot_dose,data_time,pat_data)
    return Y

def non_phys_plotter_data_model(k=[],save = False):
    if k == []:
        k = np.loadtxt('data/pat_param_drug_cpt11.txt')
    print(k)
    tot_dose = pk.CPT11_tot_dose
    pat_cp_data = pk.CPT11
    pat_sn_data = pk.SN38
    vol = pk.vol
    data_time = pk.CPT11_time
    pat_range = pk.pat_nums_cpt
    Title_str = 'CPT11'

    tmax = 36

    x0 = [0]*11
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex = True, sharey = True)
#    fig0.suptitle('CPT11 model vs data')
    fig1, ax1 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex = True, sharey = True)
#    fig1.suptitle('SN38 model vs data')
    plt.setp(ax0, xticks=np.arange(0,tmax+1,12),xticklabels = [24,12,24,12])
    plt.setp(ax1, xticks=np.arange(0,tmax+1,12),xticklabels = [24,12,24,12])
    cmax_cp = [None]*len(pat_range)
    cmax_sn = [None]*len(pat_range)
    cmax_cp_time = [None]*len(pat_range)
    cmax_sn_time = [None]*len(pat_range)
    cost = [None]*len(pat_range)
    R2 = [None]*len(pat_range)
    for i in pat_range:
        tspan = data_time[i]
        t = np.sort(list(np.linspace(0.1,36,1000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_CPT,x0,t,(k[i],tot_dose[i],vol[i]),hmax=1)
        x = np.multiply(1000,x)

        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,2],'C0--')
        ax0[int(np.floor(i/a)),i%a].plot(tspan,np.multiply(1000,pat_cp_data[i]),'.',color='C3',mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(24,1.2,'('+str(pat_range[i]+1)+')')

        ax1[int(np.floor(i/a)),i%a].plot(t,x[:,3],'C0--')
        ax1[int(np.floor(i/a)),i%a].plot(tspan,np.multiply(1000,pat_sn_data[i]),'.',color='C3',mew=2)
        ax1[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].text(24,0.03,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        plt.setp( ax1[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        
        D_cp = np.multiply(1000,pat_cp_data[i])
        D_sn = np.multiply(1000,pat_sn_data[i])
        cmax_cp[i] = np.max(x[:,2])
        cmax_sn[i] = np.max(x[:,3])
        cmax_cp_time[i] = t[np.argmax(x[:,2])]
        cmax_sn_time[i] = t[np.argmax(x[:,3])]
        M_cp = x[pos,2]
        M_sn = x[pos,3]
        cost[i] = np.nansum(np.square((M_cp - D_cp)/(2.19317759e-02)) + np.square((M_sn - D_sn)/(6.81021571e-04)))
        R2[i] = 1 - (np.nansum(np.square((M_cp - D_cp))) + np.nansum(np.square((M_sn - D_sn))))/(np.nansum((D_sn - np.nanmean(D_sn))**2) + np.nansum((D_cp - np.nanmean(D_cp))**2))

        
    if int(np.floor(len(pat_range)/2))%2 != 0:
        fig0.delaxes(ax0[-1,-1])
        fig1.delaxes(ax1[-1,-1])
    ax0[int(np.floor(i/a)),i%a].legend(labels=['model','data'],bbox_to_anchor=(1.3,1))
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(0.05,0.65,'Concentration (ng/ml)',ha='right',rotation='vertical')
    ax1[int(np.floor(i/a)),i%a].legend(labels=['model','data'],bbox_to_anchor=(2.9,1))
    fig1.text(0.5,-0.05,'Time (h)',ha='right')
    fig1.text(0.05,0.65,'Concentration (ng/ml)',ha='right',rotation='vertical')
    plt.setp(ax0,ylim=[0,np.max(cmax_cp)*1.1])
    plt.setp(ax1,ylim=[0,np.max(cmax_sn)*1.15])
    plt.show()
    if save:
        np.savetxt('cmax_cpt_patients.txt',cmax_cp)
        np.savetxt('cmax_sn_patients.txt',cmax_sn)
        np.savetxt('cmax_cp_time.txt',cmax_cp_time)
        np.savetxt('cmax_sn_time.txt',cmax_sn_time)
        folder_path = uf.new_folder('Model_fitting')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'.pdf') as pdf:
            pdf.savefig(fig0, bbox_inches='tight')
        with PdfPages(folder_path+'model_fit_for_SN38.pdf') as pdf:
            pdf.savefig(fig1, bbox_inches='tight')

    return x,cost, R2

def non_phys_plotter_data_model_colour(k=[],save = False):
    if k == []:
        k = np.loadtxt('data/pat_param_drug_cpt11.txt')
    tot_dose = pk.CPT11_tot_dose
    pat_cp_data = pk.CPT11
    pat_sn_data = pk.SN38
    data_time = pk.CPT11_time
    pat_range = pk.pat_nums_cpt
    Title_str = 'CPT11'

    tmax = 36

    x0 = [0]*9
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex = True, sharey = True)
    fig0.suptitle('CPT11 model fit to data')
    fig1, ax1 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex = True, sharey = True)
    fig1.suptitle('SN38 model fit to data')
    plt.setp(ax0, xticks=np.arange(0,tmax+1,12))
    plt.setp(ax1, xticks=np.arange(0,tmax+1,12))
    cmax_cp = [None]*len(pat_range)
    cmax_sn = [None]*len(pat_range)
    cmax_cp_time = [None]*len(pat_range)
    cmax_sn_time = [None]*len(pat_range)
    cost = [None]*len(pat_range)
    for i in pat_range:
        tspan = data_time[i]
        t = np.sort(list(np.linspace(0.1,36,1000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_CPT,x0,t,(k[i],tot_dose[i]),hmax=1)

        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,2],'--',color=(0,0,0.5))
        ax0[int(np.floor(i/a)),i%a].plot(tspan,pat_cp_data[i],'.',color=(0.7,0,0),mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(24,0.0012,'('+str(pat_range[i]+1)+')')

        ax1[int(np.floor(i/a)),i%a].plot(t,x[:,3] + x[:,7],'--',color=(0,0,0.5))
        ax1[int(np.floor(i/a)),i%a].plot(tspan,pat_sn_data[i],'.',color=(0.7,0,0),mew=2)
        ax1[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].text(24,0.00003,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        plt.setp( ax1[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        cmax_cp[i] = np.max(x[:,2])
        cmax_sn[i] = np.max(x[:,3] + x[:,7])
        cmax_cp_time[i] = t[np.argmax(x[:,2])]
        cmax_sn_time[i] = t[np.argmax(x[:,3])]
        M_cp = x[pos,2]
        M_sn = x[pos,3] + x[pos,7]
        D_cp = pat_cp_data[i]
        D_sn = pat_sn_data[i]
        cost[i] = np.nansum(np.square(M_cp - D_cp)/M_cp + np.square(M_sn - D_sn)/M_sn)

        
    if int(np.floor(len(pat_range)/2))%2 != 0:
        fig0.delaxes(ax0[-1,-1])
        fig1.delaxes(ax1[-1,-1])
    ax0[int(np.floor(i/a)),i%a].legend(labels=['model','data'],bbox_to_anchor=(1.3,1))
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(-0.05,0.65,'Concentration (mg/ml)',ha='right',rotation='vertical')
    ax1[int(np.floor(i/a)),i%a].legend(labels=['model','data'],bbox_to_anchor=(2.9,1))
    fig1.text(0.5,-0.05,'Time (h)',ha='right')
    fig1.text(-0.05,0.65,'Concentration (mg/ml)',ha='right',rotation='vertical')
    plt.setp(ax0,ylim=[0,np.max(cmax_cp)*1.1])
    plt.setp(ax1,ylim=[0,np.max(cmax_sn)*1.1])
    if save:
        np.savetxt('cmax_cpt_patients.txt',cmax_cp)
        np.savetxt('cmax_sn_patients.txt',cmax_sn)
        np.savetxt('cmax_cp_time.txt',cmax_cp_time)
        np.savetxt('cmax_sn_time.txt',cmax_sn_time)
        folder_path = uf.new_folder('Model_fitting')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'_colour.pdf') as pdf:
            pdf.savefig(fig0, bbox_inches='tight')
        with PdfPages(folder_path+'model_fit_for_SN38_colour.pdf') as pdf:
            pdf.savefig(fig1, bbox_inches='tight')

    return x,cost

def Linear_AUC():
    k = np.loadtxt('data/pooled_param_drug_cpt11.txt')    
    vol = pk.vol
    x0 = [0]*9    
    t = np.linspace(0.1,36,1000)
    doses = np.linspace(50,350,10)
    AUC = np.zeros(np.shape(doses))
    for i,dose in enumerate(doses):
        x_1 = odeint(non_phys_ode_CPT,x0,t,(k,dose,vol[0]),hmax=1)
        AUC[i] = np.trapz(x_1[:,2],t)
    fig = plt.figure()
    plt.plot(doses,AUC)
    plt.xlabel('doses (mg/m$^2$)')
    plt.ylabel('AUC (mg)')
    folder_path = uf.new_folder('Model_fitting')
    with PdfPages(folder_path+'linear_AUC_CPT11.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    return

def pooled_plotter_data_model(k=[],save = False):
    if k == []:
        k = np.loadtxt('data/pooled_param_drug_cpt11.txt')
#    print(k)
    tot_dose = pk.CPT11_tot_dose
    pat_cp_data = pk.CPT11
    pat_sn_data = pk.SN38
    vol = pk.vol
    data_time = pk.CPT11_time
    pat_range = pk.pat_nums_cpt
    Title_str = 'CPT11'

    tmax = 36

    x0 = [0]*9
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex = True, sharey = True)
#    fig0.suptitle('CPT11 model vs data')
    fig1, ax1 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex = True, sharey = True)
#    fig1.suptitle('SN38 model vs data')
    plt.setp(ax0, xticks=np.arange(0,tmax+1,12),xticklabels = [24,12,24,12])
    plt.setp(ax1, xticks=np.arange(0,tmax+1,12),xticklabels = [24,12,24,12])
    cmax_cp = [None]*len(pat_range)
    cmax_sn = [None]*len(pat_range)
    cmax_cp_time = [None]*len(pat_range)
    cmax_sn_time = [None]*len(pat_range)
    cost = [None]*len(pat_range)
    SS_tot = [None]*len(pat_range)
    SS_res = [None]*len(pat_range)
    for i in pat_range:
        tspan = data_time[i]
        t = np.sort(list(np.linspace(0.1,36,1000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_CPT,x0,t,(k,tot_dose[i],vol[i]),hmax=1)
        x = np.multiply(1000,x)

        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,2],'k--')
        ax0[int(np.floor(i/a)),i%a].plot(tspan,np.multiply(1000,pat_cp_data[i]),'.',color='0.4',mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(24,1.2,'('+str(pat_range[i]+1)+')')

        ax1[int(np.floor(i/a)),i%a].plot(t,x[:,3],'k--')
        ax1[int(np.floor(i/a)),i%a].plot(tspan,np.multiply(1000,pat_sn_data[i]),'.',color='0.4',mew=2)
        ax1[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].text(24,0.03,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        plt.setp( ax1[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        
        D_cp = np.multiply(1000,pat_cp_data[i])
        D_sn = np.multiply(1000,pat_sn_data[i])
        cmax_cp[i] = np.max(x[:,2])
        cmax_sn[i] = np.max(x[:,3])
        cmax_cp_time[i] = t[np.argmax(x[:,2])]
        cmax_sn_time[i] = t[np.argmax(x[:,3])]
        M_cp = x[pos,2]
        M_sn = x[pos,3]
        cost[i] = np.nansum(np.square((M_cp - D_cp)/(2.19317759e-02)) + np.square((M_sn - D_sn)/(6.81021571e-04)))
        SS_tot = np.nansum((D_cp - np.nanmean(D_cp))**2) + np.nansum((D_sn - np.nanmean(D_sn))**2)
        SS_res = np.nansum(np.square((M_cp - D_cp))) + np.nansum(np.square((M_sn - D_sn)))
    Cost = np.sum(cost)
    r2 = 1 - np.sum(SS_res)/np.sum(SS_tot)
    if int(np.floor(len(pat_range)/2))%2 != 0:
        fig0.delaxes(ax0[-1,-1])
        fig1.delaxes(ax1[-1,-1])
    ax0[int(np.floor(i/a)),i%a].legend(labels=['model','data'],bbox_to_anchor=(1.3,1))
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(0.05,0.65,'Concentration (ng/ml)',ha='right',rotation='vertical')
    ax1[int(np.floor(i/a)),i%a].legend(labels=['model','data'],bbox_to_anchor=(2.9,1))
    fig1.text(0.5,-0.05,'Time (h)',ha='right')
    fig1.text(0.05,0.65,'Concentration (ng/ml)',ha='right',rotation='vertical')
    plt.setp(ax0,ylim=[0,np.max(cmax_cp)*1.1])
    plt.setp(ax1,ylim=[0,np.max(cmax_sn)*1.15])
    plt.show()
    if save:
        np.savetxt('cmax_cpt_patients.txt',cmax_cp)
        np.savetxt('cmax_sn_patients.txt',cmax_sn)
        np.savetxt('cmax_cp_time.txt',cmax_cp_time)
        np.savetxt('cmax_sn_time.txt',cmax_sn_time)
        folder_path = uf.new_folder('Model_fitting')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'.pdf') as pdf:
            pdf.savefig(fig0, bbox_inches='tight')
        with PdfPages(folder_path+'model_fit_for_SN38.pdf') as pdf:
            pdf.savefig(fig1, bbox_inches='tight')

    return x,Cost, r2

if __name__ == '__main__':
    
    x,cost, R2 = non_phys_plotter_data_model(save=True)
#    r2 = np.round(np.transpose(R2),2)
#    b = np.round(cost,2)
#    print(b)
#    _, cost, r2 = pooled_plotter_data_model()
#    print('cpt11 pde SSR = ',cost)
#    print('cpt11 pde r2 = ',r2)
##    Linear_AUC()
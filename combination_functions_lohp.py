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


def non_phys_ode_LOHP(x,t,k,tot_dose,vol):
    c_b = k[0]
    ef_l = k[1]
    up_l = k[1]
    ef_o = k[2]
    up_o = k[2]
    b = k[3]
    u = k[4]

    liver_f = x[0]
    liver_b = x[1]
    Blood_f = x[2]
    Blood_b = x[3]
    organs_f = x[4]
    organs_b = x[5]
    clear_b = x[6]

    


    drug_input = dd.drug_del_LOHP

    dxdt = [ (drug_input(t,tot_dose)*0.4910 - ef_l*liver_f + up_l*Blood_f - b*liver_f + u*liver_b)/vol[0],
            ( b*liver_f - u*liver_b)/vol[0],
            (ef_l*liver_f - up_o*Blood_f - up_l*Blood_f - c_b*Blood_f - b*Blood_f + u*Blood_b )/vol[1],
            ( b*Blood_f - u*Blood_b )/vol[1],
            (up_o*Blood_f - ef_o*organs_f - b*organs_f + u*organs_b)/vol[2],
            ( b*organs_f - u*organs_b)/vol[2],
            c_b*Blood_f]
    return dxdt

def pooled_data_cost(k):
    pat_range = pk.pat_nums_lohp
    cost = 0
    for j in range(0,len(pat_range)):
        pat_data = [None]*2
        tot_dose = pk.LOHP_tot_dose[pat_range[j]]
        pat_data[0] = pk.LOHP_free[pat_range[j]]
        pat_data[1] = pk.LOHP_total[pat_range[j]]
        vol = pk.vol[pat_range[j]]
        data_time = pk.LOHP_time[pat_range[j]]
        cost += non_phys_param_cost_single_lohp(k,tot_dose,data_time,pat_data,vol)
    return cost

def non_phys_param_cost_single_lohp(k,pat_dose,data_time,pat_data,vol):
    cost = 0
    x0 = [0,0,0,0,0,0,0]
    tspan = data_time
    t = np.sort(list(np.linspace(9,36,10000))+list(tspan))
    t = np.sort(list(set(t)))
    pos = np.where(np.in1d(t,tspan))[0]
    x = odeint(non_phys_ode_LOHP,x0,t,(k,pat_dose,vol),hmax=1)
    M_u = x[pos,2]
    D_u = pat_data[0]
    M_t = x[pos,3]+x[pos,2]
    D_t = pat_data[1]
    
    tot_bound = (vol[0]*x[-1,1] + vol[1]*x[-1,3] + vol[2]*x[-1,5])
    O_b_cost = 100*(vol[2]*x[-1,5]/tot_bound - 0.84)**2
    L_b_cost = 100*(vol[0]*x[-1,1]/tot_bound - 0.12)**2
    cleared = 100*(x[-1,-1]/pat_dose - 0.54)**2
    cost = np.nansum(np.square((M_u - D_u)/(1.48786696e-05)) + np.square((M_t - D_t)/(2.6972e-05))) + O_b_cost + L_b_cost + cleared
    return cost


def evaluate_pk_cost(values,tot_dose,data_time,pat_data):
    Y = np.zeros([values.shape[0]])
    for i, X in enumerate(values):
        Y[i] = non_phys_param_cost_single_lohp(X,tot_dose,data_time,pat_data)
    return Y

def non_phys_plotter_data_model(k=[],save = False):
    if k == []:
        k = np.loadtxt('data/pat_param_drug_lohp.txt')
        print(k)

    tot_dose = pk.LOHP_tot_dose
    pat_free_data = pk.LOHP_free
    data_time = pk.LOHP_time
    pat_tot_data = pk.LOHP_total
    pat_data = [None]*2
    pat_data[0] = pk.LOHP_free[0]
    pat_data[1] = pk.LOHP_total[0]
    vol = pk.vol
    data_time = pk.LOHP_time
    pat_range = pk.pat_nums_lohp
    Title_str = 'LOHP'
#    print(non_phys_param_cost_single_lohp(k[0],tot_dose[0],data_time[0],pat_data))
    x0 = [0,0,0,0,0,0,0]

#    ymax_f = np.nanmax(pat_free_data)*1.08
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex=True,sharey=True)
#    fig0.suptitle(Title_str+' total, model vs data')
    plt.setp(ax0,xlim=[9,37], xticks=np.arange(9,37,12),xticklabels=np.arange(9,37,12)%24)
    fig1, ax1 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex=True,sharey=True)
    plt.setp(ax1, xlim=[9,37], xticks=np.arange(9,37,12),xticklabels=np.arange(9,37,12)%24)#,ylim=[0,ymax_f])
#    fig1.suptitle(Title_str+' free, model vs data')
    cmax_f = [None]*len(pat_range)
    cmax_time = [None]*len(pat_range)
    cost = [None]*len(pat_range)
    R2 = [None]*len(pat_range)
    for i in pat_range:
        tspan = data_time[i]
        t = np.sort(list(np.linspace(0.1,36,10000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_LOHP,x0,t,(k[i],tot_dose[i],vol[i]),hmax=0.25)
        x=np.multiply(x,1000)
        
        
        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,3]+x[:,2],'C0--')
        ax0[int(np.floor(i/a)),i%a].plot(tspan,np.multiply(1000,pat_tot_data[i]),'.',color='C3',mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(9.5,1.2,'('+str(pat_range[i]+1)+')')
        
        
        ax1[int(np.floor(i/a)),i%a].plot(t,x[:,2],'C0--')
        ax1[int(np.floor(i/a)),i%a].plot(tspan,np.multiply(1000,pat_free_data[i]),'.',color='C3',mew=2)
        ax1[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].text(9.5,0.47,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        plt.setp( ax1[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        
        data_tot = np.multiply(1000,pat_tot_data[i])
        data_free = np.multiply(1000,pat_free_data[i])
        cmax_f[i] = np.max(x[:,2])
        cmax_time[i] = t[np.argmax(x[:,2])]
        M_u = x[pos,2]
        D_u = data_free
        M_t = x[pos,3]+x[pos,2]
        D_t = data_tot
        cost[i] = np.nansum(np.square((M_u - D_u)/(1.48786696e-02)) + np.square((M_t - D_t)/(2.6972e-02)))
        R2[i] = 1 - (np.nansum(np.square(M_u - D_u)) + np.nansum(np.square((M_t - D_t))))/(np.nansum((D_t - np.nanmean(D_t))**2) + np.nansum((D_u - np.nanmean(D_u))**2))



    if int(np.floor(len(pat_range)/2)) != a:
        fig0.delaxes(ax0[-1,-1])
        fig1.delaxes(ax1[-1,-1])
    ax0[int(np.floor(i/6)),i%6].legend(labels=['model','data'],bbox_to_anchor=(3.5,1))
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(0.05,0.3,'Concentration (ng/ml)',ha='right',rotation='vertical')
    ax1[int(np.floor(i/6)),i%6].legend(['model','data'],bbox_to_anchor=(3.5,1))
    fig1.text(0.5,-0.05,'Time (h)',ha='right')
    fig1.text(0.05,0.3,'Concentration (ng/ml)',ha='right',rotation='vertical')
    if save:
        np.savetxt('cmax_lohp_patient.txt',cmax_f)
        np.savetxt('cmax_lohp_time.txt',cmax_time)
        folder_path = uf.new_folder('Model_fitting')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'_total.pdf') as pdf:
            pdf.savefig(fig0, bbox_inches='tight')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'_free.pdf') as pdf:
            pdf.savefig(fig1, bbox_inches='tight')

    return x,cost, R2

def non_phys_plotter_data_model_colour(k=[],save = False):
    if k == []:
        k = np.loadtxt('data/pat_param_drug_lohp.txt')
        #print(k)

    tot_dose = pk.LOHP_tot_dose
    pat_free_data = pk.LOHP_free
    data_time = pk.LOHP_time
    pat_tot_data = pk.LOHP_total
    data_time = pk.LOHP_time
    pat_range = pk.pat_nums_lohp
    vol = pk.vol
    Title_str = 'LOHP'



    
    x0 = [0,0,0,0,0,0]


    ymax_f = np.nanmax(pat_free_data)*1.08
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex=True,sharey=True)
    fig0.suptitle('LOHP total, model fit to data')
    plt.setp(ax0,xlim=[9,37], xticks=np.arange(9,37,12),xticklabels=np.arange(9,37,12)%24)
    fig1, ax1 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex=True,sharey=True)
    plt.setp(ax1, xlim=[9,37], xticks=np.arange(9,37,12),xticklabels=np.arange(9,37,12)%24,ylim=[0,ymax_f])
    fig1.suptitle('LOHP free, model fit to data')
    cmax_f = [None]*len(pat_range)
    cmax_time = [None]*len(pat_range)
    cost = [None]*len(pat_range)

    for i in pat_range:
        tspan = data_time[i]
        t = np.sort(list(np.linspace(0.1,36,10000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_LOHP,x0,t,(k[i],tot_dose[i],vol[i]),hmax=0.25)

        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,3]+x[:,2],'--',color=(0,0,0.5))
        ax0[int(np.floor(i/a)),i%a].plot(tspan,pat_tot_data[i],'.',color=(0.7,0,0),mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(9.5,0.0012,'('+str(pat_range[i]+1)+')')

        ax1[int(np.floor(i/a)),i%a].plot(t,x[:,2],'--',color=(0,0,0.5))
        ax1[int(np.floor(i/a)),i%a].plot(tspan,pat_free_data[i],'.',color=(0.7,0,0),mew=2)
        ax1[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].text(9.5,0.00047,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        plt.setp( ax1[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )

        cmax_f[i] = np.max(x[:,2])
        cmax_time[i] = t[np.argmax(x[:,2])]
        M_u = x[pos,2]
        D_u = pat_free_data[i]
        M_t = x[pos,3]+x[pos,2]
        D_t = pat_tot_data[i]
        cost[i] = np.nansum(np.square(M_u - D_u)/(0.1*np.max(D_u)) + np.square(M_t - D_t)/(0.1*np.max(D_t)))



    if int(np.floor(len(pat_range)/2)) != a:
        fig0.delaxes(ax0[-1,-1])
        fig1.delaxes(ax1[-1,-1])
    ax0[int(np.floor(i/6)),i%6].legend(labels=['model','data'],bbox_to_anchor=(3.5,1))
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(-0.05,0.7,'Concentration (mg/ml)',ha='right',rotation='vertical')
    ax1[int(np.floor(i/6)),i%6].legend(['model','data'],bbox_to_anchor=(3.5,1))
    fig1.text(0.5,-0.05,'Time (h)',ha='right')
    fig1.text(-0.05,0.7,'Concentration (mg/ml)',ha='right',rotation='vertical')
    if save:
        np.savetxt('cmax_lohp_patient.txt',cmax_f)
        np.savetxt('cmax_lohp_time.txt',cmax_time)
        folder_path = uf.new_folder('Model_fitting')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'_total_colour.pdf') as pdf:
            pdf.savefig(fig0, bbox_inches='tight')
        with PdfPages(folder_path+'model_fit_for_'+Title_str+'_free_colour.pdf') as pdf:
            pdf.savefig(fig1, bbox_inches='tight')

    return x,cost

def plot_pooled_data(k=None):
    if k == None:
        k = np.loadtxt('data/pooled_param_drug_lohp.txt')
        #print(k)

    tot_dose = pk.LOHP_tot_dose
    pat_free_data = pk.LOHP_free
    data_time = pk.LOHP_time
    pat_tot_data = pk.LOHP_total
    data_time = pk.LOHP_time
    pat_range = pk.pat_nums_lohp
    vol = pk.vol
    Title_str = 'LOHP'



    
    x0 = [0,0,0,0,0,0,0,0]


    ymax_f = np.nanmax(pat_free_data)*1.08
    a = int(np.ceil(len(pat_range)/2))
    fig0, ax0 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex=True,sharey=True)
    fig0.suptitle('LOHP total, model fit to data')
    plt.setp(ax0,xlim=[9,37], xticks=np.arange(9,37,12),xticklabels=np.arange(9,37,12)%24)
    fig1, ax1 = plt.subplots(2,int(np.ceil(len(pat_range)/2)),sharex=True,sharey=True)
    plt.setp(ax1, xlim=[9,37], xticks=np.arange(9,37,12),xticklabels=np.arange(9,37,12)%24,ylim=[0,ymax_f])
    fig1.suptitle('LOHP free, model fit to data')
    
    cost = [None]*len(pat_range)
    SS_tot = [None]*len(pat_range)
    SS_res = [None]*len(pat_range)

    for i in pat_range:
        tspan = data_time[i]
        t = np.sort(list(np.linspace(0.1,36,10000))+list(tspan))
        t = np.sort(list(set(t)))
        pos = np.where(np.in1d(t,tspan))[0]
        x = odeint(non_phys_ode_LOHP,x0,t,(k,tot_dose[i],vol[i]),hmax=0.25)

        ax0[int(np.floor(i/a)),i%a].plot(t,x[:,3]+x[:,2],'--',color=(0,0,0.5))
        ax0[int(np.floor(i/a)),i%a].plot(tspan,pat_tot_data[i],'.',color=(0.7,0,0),mew=2)
        ax0[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax0[int(np.floor(i/a)),i%a].text(9.5,0.0012,'('+str(pat_range[i]+1)+')')

        ax1[int(np.floor(i/a)),i%a].plot(t,x[:,2],'--',color=(0,0,0.5))
        ax1[int(np.floor(i/a)),i%a].plot(tspan,pat_free_data[i],'.',color=(0.7,0,0),mew=2)
        ax1[int(np.floor(i/a)),i%a].spines['right'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].spines['top'].set_visible(False)
        ax1[int(np.floor(i/a)),i%a].text(9.5,0.00047,'('+str(pat_range[i]+1)+')')

        plt.setp( ax0[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )
        plt.setp( ax1[int(np.floor(i/a)),i%a].xaxis.get_majorticklabels(), rotation=90 )


        M_u = x[pos,2]
        D_u = pat_free_data[i]
        M_t = x[pos,3]+x[pos,2]
        D_t = pat_tot_data[i]
        cost[i] = np.nansum(np.square((M_u - D_u)/(1.48786696e-02)) + np.square((M_t - D_t)/(2.6972e-02)))
        SS_tot[i] = np.nansum((D_u - np.nanmean(D_u))**2) + np.nansum((D_t - np.nanmean(D_t))**2)
        SS_res[i] = np.nansum(np.square(M_u - D_u)) + np.nansum(np.square((M_t - D_t)))
        
    Cost = np.sum(cost)
    r2 = 1 - np.sum(SS_res)/np.sum(SS_tot)



    if int(np.floor(len(pat_range)/2)) != a:
        fig0.delaxes(ax0[-1,-1])
        fig1.delaxes(ax1[-1,-1])
    ax0[int(np.floor(i/6)),i%6].legend(labels=['model','data'],bbox_to_anchor=(3.5,1))
    fig0.text(0.5,-0.05,'Time (h)',ha='right')
    fig0.text(-0.05,0.7,'Concentration (mg/ml)',ha='right',rotation='vertical')
    ax1[int(np.floor(i/6)),i%6].legend(['model','data'],bbox_to_anchor=(3.5,1))
    fig1.text(0.5,-0.05,'Time (h)',ha='right')
    fig1.text(-0.05,0.7,'Concentration (mg/ml)',ha='right',rotation='vertical')
    
    folder_path = uf.new_folder('Model_fitting')
    with PdfPages(folder_path+'model_fit_for_pooled'+Title_str+'_total_colour.pdf') as pdf:
        pdf.savefig(fig0, bbox_inches='tight')
    with PdfPages(folder_path+'model_fit_for_pooled'+Title_str+'_free_colour.pdf') as pdf:
        pdf.savefig(fig1, bbox_inches='tight')

    return x,Cost, r2



if __name__ == '__main__':

    x, cost, R2 = non_phys_plotter_data_model()
#    r2 = np.round(np.transpose(R2),2)
#    b=np.round(cost,2)
#    print(b)
#    _, cost, R2 = plot_pooled_data()
    print('lohp pde SSR = ',cost)
    print('lohp pde r2 = ',R2)
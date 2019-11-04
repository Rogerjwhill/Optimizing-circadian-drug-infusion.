
"""
Created on Fri Oct  6 10:03:22 2017

@author: roger
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import Useful_functions as uf

file_path = uf.new_folder('PK_data')
# surface area of patients
m_sq = [1.81,1.56,1.62,2.00,1.79,1.99,2.13,1.85,1.58,1.91,1.95]
# average volumes previously used: 
#vol = [1057,5100,73900]
vol = [[1539,4737,65289],
       [1236,3461,53139],
       [1224,3658,50614],
       [1798,4819,77765],
       [1462,4636,61303],
       [1690,5327,70900],
       [1872,5832,78914],
       [1537,4861,64437],
       [1208,3517,51124],
       [1618,4990,68479],
       [1619,5238,67249]]

# total dose = mg/m^2 times m^2
CPT11_tot_dose = np.multiply(180,m_sq)

CPT11_time = [[2,4,5,6,8.5,10.25,33.75],
              [2,4,5,6,8,10.25,33.75],
              [2,4,5,5.93,7.97,10.25,33.75],
              [2,4,5,6,8.03,10.25,33.75],
              [2.02,4,5,6,8,10.25,33.75],
              [2,4,5,6,8,10.17,33.75],
              [2,4,5,6,8.03,10.25,33.75],
              [2,4,5,6,8,10.25,33.75],
              [2,4,5,6,8,10.25,33.75],
              [2,4,5,6,8,10.25,33.75],
              [2,4,5,6,8,10.25,33.75]]
#ng/ml
CPT11 = [[0,313.39e-06,758.78e-06,978.22e-06,753.97e-06,622.54e-06,54.58e-06],
         [0,195.83e-06,624.45e-06,805.17e-06,845.08e-06,657.52e-06,83.69e-06],
         [0,209.26e-06,477.86e-06,622.50e-06,537.17e-06,303.59e-06,15.55e-06],
         [0,0.00,np.nan,812.97e-06,649.61e-06,610.27e-06,21.71e-06],
         [0,298.43e-06,710.65e-06,1220.86e-06,839.86e-06,1055.09e-06,109.49e-06],
         [0,20.25e-06,396.90e-06,810.78e-06,799.32e-06,584.62e-06,69.21e-06],
         [0,241.51e-06,555.25e-06,803.03e-06,766.99e-06,607.06e-06,35.97e-06],
         [0,168.27e-06,400.67e-06,546.09e-06,429.83e-06,332.80e-06,18.96e-06],
         [0,76.42e-06,348.56e-06,605.45e-06,504.06e-06,444.59e-06,34.65e-06],
         [0,302.05e-06,784.35e-06,1035.03e-06,931.34e-06,895.94e-06,98.67e-06],
         [0,0.00,509.37e-06,882.61e-06,841.69e-06,np.nan,np.nan]]


SN38 =  [[0,6.11e-06,9.33e-06,12.04e-06,8.52e-06,6.15e-06,1.74e-06],
         [0,5.91e-06,14.65e-06,10.46e-06,16.57e-06,11.20e-06,3.92e-06],
         [0,10.06e-06,13.49e-06,14.76e-06,16.21e-06,7.95e-06,2.60e-06],
         [0,0.00,np.nan,22.29e-06,17.62e-06,12.78e-06,0.00e-06],
         [0,7.92e-06,18.25e-06,34.16e-06,25.57e-06,35.86e-06,3.73e-06],
         [0,0.00e-06,7.59e-06,11.26e-06,8.29e-06,5.12e-06,2.28e-06],
         [0,3.24e-06,11.12e-06,12.79e-06,10.85e-06,9.11e-06,1.37e-06],
         [0,3.92e-06,10.61e-06,14.38e-06,11.86e-06,8.66e-06,2.65e-06],
         [0,0.00e-06,8.75e-06,12.19e-06,10.58e-06,6.85e-06,0.00e-06],
         [0,6.86e-06,17.36e-06,22.33e-06,28.70e-06,19.93e-06,5.03e-06],
         [0,0.00,10.38e-06,13.82e-06,12.21e-06,np.nan,np.nan]]
pat_nums_cpt = [0,1,2,3,4,5,6,7,8,9,10]
# pateint ID
CPT11_pat_id = ['REN-JA','VIAMA','MORCA','BOMA','COUJE','LEGMI','FRENCH','DAOAB','POICO','PLAPH','PAPAL']
# total dose = mg/m^2 times m^2
FU_tot_dose = np.multiply(933,m_sq)
FU_time = [[21.75,25.25,28,31.25,33.75],
           [21.83,25.17,27.83,31.25,33.75],
           [21.8,25.25,28,31.33,33.75],
           [21.78,25.25,28,31.25,33.75],
           [21.75,25.25,28,31.25,33.83],
           [22,25.25,28.03,31.25,33.75],
           [21.75,25.25,28,31.25,33.75],
           [21.75,25.25,28,31.25,33.75],
           [21.75,25.25,28,31.25,33.83],
           [21.75,25.25,28,31.25,33.75],
           [21.75,25.25,28,31.25,33.75]]
#ng/ml
FU = [[0,0,587e-06,212e-06,0],
    [0,233e-06,1058e-06,581e-06,0],
    [0,0,845e-06,165e-06,0],
    [0,86e-06,427e-06,81e-06,0],
    [0,0,804e-06,290e-06,318e-06],
    [np.nan]*5,
    [0,73e-06,372e-06,89e-06,0],
    [0,0,612e-06,161e-06,0],
    [0,0,524e-06,83e-06,97e-06],
    [0,0,1641e-06,761e-06,101e-06],
    [np.nan]*5]
pat_nums_fu =[0,1,2,3,4,6,7,8,9]
# pateint ID
FU_pat_id = ['REN-JA','VIAMA','MORCA','BOMA','COUJE','FRENCH','DAOAB','POICO','PLAPH']
# total dose = mg/m^2 times m^2
LOHP_tot_dose = np.multiply(28,m_sq)
LOHP_time = [[10.25,13.25,16.25,19.25,22,28],
             [10.25,13.25,16.25,19.25,22,28],
             [10.25,13.25,16.35,19.22,22,27.84],
             [10.25,13.25,16.25,19.25,21.75,28],
             [10.25,13.08,16.25,19.25,21.78,28],
             [10.25,13.25,16.25,19.25,21.75,28],
             [10.17,13.23,16.33,19.27,22,28],
             [10.25,13.32,16.27,19.33,21.75,28.03],
             [10.25,13.25,16.41,19.3,21.75,28],
             [10.25,13.25,16.41,19.3,21.75,28],
             [10.25,13.25,16.41,19.3,21.75,28]]
#ng/ml
LOHP_total = [[0,0,538.53e-06,776.43e-06,872.45e-06,782.89e-06],
            [0e-06,0e-06,458.76e-06,700.95e-06,1106.58e-06,746.33e-06],
            [0,0,592.42e-06,1092.19e-06,1022.75e-06,881.71e-06],
            [0,0,626.29e-06,1018.02e-06,896.30e-06,745.74e-06],
            [0,0,627.51e-06,1025.22e-06,963.34e-06,999.94e-06],
            [0,0,259.70e-06,474.74e-06,520.50e-06,579.14e-06],
            [0,0,497.11e-06,283.27e-06,1153.35e-06,	621.86e-06],
            [0,0,478.07e-06,743.49e-06,720.06e-06,766.29e-06],
            [0,0,701.11e-06,1196.35e-06,1052.57e-06,938.74e-06],
            [0,0,471.60e-06,768.06e-06,729.45e-06,819.67e-06],
            [np.nan]*6]
#ng/ml
LOHP_free = [[0.00,0.00,49.37e-06,54.57e-06,182.23e-06,90.39e-06],
            [0.00,31.89e-06,129.80e-06,119.51e-06,283.28e-06,120.35e-06],
            [0.00,0.00,131.52e-06,123.57e-06,439.64e-06,73.30e-06],
            [0.00,0.00,203.15e-06,np.nan,93.40e-06,56.56e-06],
            [0.00,0.00,168.20e-06,125.77e-06,129.72e-06,106.93e-06],
            [0.00,0.00,92.55e-06,73.46e-06,55.04e-06,42.54e-06],
            [0.00,0.00,165.41e-06,58.73e-06,504.03e-06,90.48e-06],
            [0.00,0.00,np.nan,np.nan,85.37e-06,	79.71e-06],#I have added nans here to see if it effects the fitting process
            [0,0.00,228.15e-06,207.50e-06,173.71e-06,179.12e-06],
            [0,0.00,109.42e-06,104.54e-06,85.05e-06,69.50e-06],
            [np.nan]*6]
pat_nums_lohp = [0,1,2,3,4,5,6,7,8,9]
# pateint ID
LOHP_pat_id = ['REN-JA','VIAMA','MORCA','BOMA','COUJE','LEGMI','FRENCH','DAOAB','POICO','PLAPH']

def plot_LOHP(save=False):
    ticks = np.arange(9,30,1)
    fig = plt.figure()
    plt.suptitle('Total Platinum over time')
    ax = plt.subplot2grid((4,1),(0,0),rowspan=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks%24)
    ax1 = plt.subplot2grid((4,1),(3,0),sharex=ax)
    for i in range(len(LOHP_total)):
        ax.plot(LOHP_time[i],LOHP_total[i],'--x',label='pat='+str(i+1))
    ax.set_xlim([9,30])
    ax.legend(bbox_to_anchor=(1.25, 1.02))
    ax.tick_params(axis='x',direction='in')
    plt.xlabel('Clock Time')
    ax.set_ylabel('Concentration')


    t = np.arange(10.25,21.75,0.1)
    y = (1+np.cos(2*np.pi/11.5*(16-t)))
    ax1.plot(t,y)
    ax1.set_ylim([0,2.1])
    ax1.get_yaxis().set_ticks([])
    plt.ylabel('delivery\nschedule',multialignment='center')
    fig.subplots_adjust(hspace=0)

    with PdfPages(file_path+'Pk_data_pres_tot.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    fig = plt.figure()
    plt.suptitle('Free Platinum over time')
    ax = plt.subplot2grid((4,1),(0,0),rowspan=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks%24)
    ax.tick_params(axis='x',direction='in')
    ax1 = plt.subplot2grid((4,1),(3,0),sharex=ax)
    for i in range(len(LOHP_free)):
        ax.plot(LOHP_time[i],LOHP_free[i],'--x',label='pat='+str(i+1))
    ax.set_xlim([9,30])
    ax.legend(bbox_to_anchor=(1.25, 1.02))
    plt.xlabel('Clock Time')
    ax.set_ylabel('Concentration')


    t = np.arange(10.25,21.75,0.1)
    y = (1+np.cos(2*np.pi/11.5*(16-t)))
    ax1.plot(t,y)
    ax1.set_ylim([0,2.1])
    ax1.get_yaxis().set_ticks([])
    plt.ylabel('delivery\nschedule',multialignment='center')
    fig.subplots_adjust(hspace=0)
    if save:
        with PdfPages(file_path+'Pk_data_pres_free.pdf') as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            
            
def plot_CPT11(save=False):
    ticks = np.arange(0,34,4)
    lims = [0,34]
    fig = plt.figure()
    plt.suptitle('CPT11 over time')
    ax = plt.subplot2grid((4,1),(0,0),rowspan=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks%24)
    ax1 = plt.subplot2grid((4,1),(3,0),sharex=ax)
    for i in range(len(CPT11_tot_dose)):
        ax.plot(CPT11_time[i],CPT11[i],'--x',label='pat='+str(i+1))
    ax.set_xlim(lims)
    ax.legend(bbox_to_anchor=(1.25, 1.02))
    ax.tick_params(axis='x',direction='in')
    plt.xlabel('Clock Time')
    ax.set_ylabel('Concentration')


    t = np.arange(2,8,0.1)
    y = (1+np.cos(2*np.pi/6*(5-t)))
    ax1.plot(t,y)
    ax1.set_ylim([0,2.1])
    ax1.get_yaxis().set_ticks([])
    plt.ylabel('delivery\nschedule',multialignment='center')
    fig.subplots_adjust(hspace=0)
    if save:
        with PdfPages(file_path+'Pk_data_pres_cpt11.pdf') as pdf:
            pdf.savefig(fig, bbox_inches='tight')

    fig = plt.figure()
    plt.suptitle('SN38 over time')
    ax = plt.subplot2grid((4,1),(0,0),rowspan=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks%24)
    ax.tick_params(axis='x',direction='in')
    ax1 = plt.subplot2grid((4,1),(3,0),sharex=ax)
    for i in range(len(CPT11)):
        ax.plot(CPT11_time[i],SN38[i],'--x',label='pat='+str(i+1))
    ax.set_xlim(lims)
    ax.legend(bbox_to_anchor=(1.25, 1.02))
    plt.xlabel('Clock Time')
    ax.set_ylabel('Concentration')

    ax1.plot(t,y)
    ax1.set_ylim([0,2.1])
    ax1.get_yaxis().set_ticks([])
    plt.ylabel('delivery\nschedule',multialignment='center')
    fig.subplots_adjust(hspace=0)
    if save:
        with PdfPages(file_path+'Pk_data_sn38.pdf') as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            
def plot_FU(save=False):
    ticks = np.arange(21,35,2)
    lims = [21,35]
    fig = plt.figure()
    plt.suptitle('5-FU over time')
    ax = plt.subplot2grid((4,1),(0,0),rowspan=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks%24)
    ax1 = plt.subplot2grid((4,1),(3,0),sharex=ax)
    for i in range(len(FU)):
        ax.plot(FU_time[i],FU[i],'--x',label='pat='+str(i+1))
    ax.set_xlim(lims)
    ax.legend(bbox_to_anchor=(1.25, 1.02))
    ax.tick_params(axis='x',direction='in')
    plt.xlabel('Clock Time')
    ax.set_ylabel('Concentration')


    t = np.arange(22,34,0.1)
    y = (1+np.cos(2*np.pi/11.5*(28-t)))
    ax1.plot(t,y)
    ax1.set_ylim([0,2.1])
    ax1.get_yaxis().set_ticks([])
    plt.ylabel('delivery\nschedule',multialignment='center')
    fig.subplots_adjust(hspace=0)
    if save:
        with PdfPages(file_path+'Pk_data_pres_fu.pdf') as pdf:
            pdf.savefig(fig, bbox_inches='tight')

    




if __name__ == '__main__':
#    print(np.nanstd(FU,axis=0))
#    print(np.nanstd(LOHP_total,axis=0))
#    print(np.nanstd(LOHP_free,axis=0))
#    print(np.nanstd(CPT11,axis=0))
#    print(np.nanstd(SN38,axis=0))
    
    plot_CPT11() 
    plot_FU()
    plot_LOHP()

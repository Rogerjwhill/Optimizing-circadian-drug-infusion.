
"""
Created on Thu Oct  5 14:50:38 2017

@author: roger
"""
import numpy as np
import matplotlib.pyplot as plt
#import Useful_functions as uf

def drug_del_5_fu(t,total_dose=1):

    fin = np.mod(t,24*14) < (96+22+12) # 4days + start time + length of daily treatment
    S = np.mod(t,24*14) > 22
    Total_scale = np.multiply(fin,S)
    daily0 = np.mod(t,24) > 0.55
    daily1 = np.mod(t,24) < 9.5
    daily2 = np.mod(t,24) > 9.75
    daily3 = np.mod(t,24) < 10
    D0 = np.multiply(daily0,daily1)
    D1 = np.multiply(daily2,daily3)
    f_out0 = np.multiply(D0,1/11.5*(1+np.cos(np.pi/5.75*(4 - np.mod(t,24))))*total_dose)
    f_out1 = np.multiply(D1,0.2*(1+np.cos(np.pi*4*(16 - np.mod(t,24))))*total_dose)
    f_out = np.multiply(Total_scale,(f_out0+f_out1))


    return f_out

def drug_del_5_fu_orig(t,total_dose=1):
    # 11.5 hour start at 22.75 finish at 9.75
    daily0 = np.mod(t,24) > 22.25
    daily1 = np.mod(t,24) < 9.75
    D0 = np.multiply(daily0, 1/11.5*(1+np.cos(np.pi/5.75*(28 - np.mod(t,24))))*total_dose)
    D1 = np.multiply(daily1, 1/11.5*(1+np.cos(np.pi/5.75*(4 - np.mod(t,24))))*total_dose)
    fin = np.mod(t,24*14) < (96+22+12) # 4days + start time + length of daily treatment
    S = np.mod(t,24*14) > 22
    Total_scale = np.multiply(fin,S)
    f_out0 = D0 + D1
    f_out = np.multiply(Total_scale,(f_out0))
    
    return f_out

def drug_del_CPT11(t,tot_dose=1):
    L0 = t >=2.863
    L1 = t <= 8
    G0 = t >=9.75
    G1 = t <= 10
    L = np.multiply(L0,L1)
    G = np.multiply(G0,G1)
    f_out0 = np.multiply(L,1/6*(1 + np.cos((np.pi/3) *(5 - t)))*tot_dose)
    f_out1 = np.multiply(G,0.45/6*(1+np.cos(np.pi*8*(9.875 - np.mod(t,24))))*tot_dose)
    f_out = f_out0 + f_out1
    return f_out
    

def drug_del_CPT11_orig(t,tot_dose=1):
    L0 = t >=2
    L1 = t <= 8
    L = np.multiply(L0,L1)
    f_out = np.multiply(L,1/6*(1 + np.cos((np.pi/3) *(5 - t)))*tot_dose)
    return f_out

def drug_del_LOHP(t,total_dose=1):
    fin = np.mod(t,24*14) < (96+10+12) # 4days + start time + length of daily treatment
    S = np.mod(t,24*14) > 10
    Total_scale = np.multiply(fin,S)
    daily0 = np.mod(t,24) > 13.6
    daily1 = np.mod(t,24) < 21.5
    daily2 = np.mod(t,24) > 21.75
    daily3 = np.mod(t,24) < 22.01
    D0 = np.multiply(daily0,daily1)
    D1 = np.multiply(daily2,daily3)
    f_out0 = np.multiply(D0,1/11*(1+np.cos(np.pi/5.75*(16 - np.mod(t,24))))*total_dose)
    f_out1 = np.multiply(D1,0.4*(1+np.cos(np.pi*4*(16 - np.mod(t,24))))*total_dose)
    f_out = np.multiply(Total_scale,(f_out0+f_out1))

    return f_out

def drug_del_LOHP_orig(t,total_dose=1):
    fin = np.mod(t,24*14) < (96+10+12) # 4days + start time + length of daily treatment
    S = np.mod(t,24*14) > 10
    Total_scale = np.multiply(fin,S)
    daily0 = np.mod(t,24) > 10.25
    daily1 = np.mod(t,24) < 21.75
    D0 = np.multiply(daily0,daily1)
    f_out0 = np.multiply(D0,1/11.5*(1+np.cos(np.pi/5.75*(16 - np.mod(t,24))))*total_dose)

    f_out = np.multiply(Total_scale,f_out0)

    return f_out

if __name__ == '__main__':
    t = np.arange(0,11,0.001)
#    drug5Fu = drug_del_5_fu_orig(t)
    drugCPT11 = drug_del_CPT11(t)
#    drugLOHP = drug_del_LOHP_orig(t)
#    plt.plot(t,drug5Fu)
    plt.plot(t,drugCPT11)
#    plt.plot(t,drugLOHP)
#    plt.xlim([24,35])
    print(np.trapz(drugCPT11,t))
    # this should be 1


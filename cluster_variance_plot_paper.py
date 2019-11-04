"""
Created on Fri Apr  5 10:38:12 2019

@author: roger
"""
import numpy as np
from matplotlib import pyplot as plt
import drug_data as dd
import Useful_functions as uf
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mtick

pat_num = [dd.pat_nums_cpt,dd.pat_nums_lohp,dd.pat_nums_fu]
drug_names = ['_cpt11','_lohp','_5fu']
titles = ['CPT11 ','L-OHP ','5-FU ']
point_style = ['o','s','^','>','D']
point_color = ['C0','C1','C3','C2','C4','C5','C6','C7','C8']
labels = [['Clearance$_{cpt,O/L}$','Clearance$_{cpt,B}$','Clearance$_{sn,O/L}$','Clearance$_{sn,B}$','Bioactivation$_{cpt}$','Uptake/Efflux$_{O/L}$'],['Clearance$_{O/L}$','Clearance$_B$','Efflux/Uptake$_L$','Efflux/Uptake$_O$','Bind','Unbind'],['Clearance$_{O/L}$','Clearance$_B$','Uptake/Efflux$_{O/L}$']]
metric = ['euclidean']
cluster_nums = [4,9,4]
fig = plt.figure(figsize=(12,12))
for i in [0,1,2]:
    x = [range(1,7),range(1,6),range(1,4)]
    k = np.loadtxt('data/pat_param_drug'+drug_names[i]+'.txt')
    cluster_membership = np.load('data/cluster_membership'+drug_names[i]+'.npy')
    print(cluster_membership)
    
    plt.subplot(2,3,i+1)
    for j in range(len(k)):
        plt.plot(x[i],k[j],'-x',markersize=8,markeredgecolor='k', marker=point_style[cluster_membership[j]] ,color = point_color[cluster_membership[j]])
    plt.yscale('log')
#    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
    plt.xticks(x[i],labels[i],rotation=-90)
#    plt.yticks(rotation=90)
    plt.ylim([np.min(k)*0.5,np.max(k)*1.05])
    plt.legend
    if i==0:
        plt.ylabel('parameter values (ml/h) (mg/h)*')

    cluster_membership = np.load('data/cluster_membership'+drug_names[i]+'.npy')
    pos = np.load('MDS_points'+drug_names[i]+'.npy')

    plt.subplot(2,3,i+4)
    for j in range(len(k)):
        plt.plot((pos[j, 0]- min(pos[:,0]))/(max(pos[:,0])-min(pos[:,0])), (pos[j, 1]- min(pos[:,1]))/(max(pos[:,1])-min(pos[:,1])) ,markersize=8,markeredgecolor='k',marker=point_style[cluster_membership[j]],color=point_color[cluster_membership[j]])
    for n in range(cluster_nums[i]):
        plt.plot((pos[j+n+1, 0]- min(pos[:,0]))/(max(pos[:,0])-min(pos[:,0])), (pos[j+n+1, 1]- min(pos[:,1]))/(max(pos[:,1])-min(pos[:,1])),'x',markersize=10,markeredgecolor='k',markerfacecolor=point_color[n])
#    
    x = (pos[:, 0]- min(pos[:,0]))/(max(pos[:,0])-min(pos[:,0])) + 0.02
    y = (pos[:, 1]- min(pos[:,1]))/(max(pos[:,1])-min(pos[:,1]))
#    for j, txt in enumerate(pos[0:-cluster_nums[i]]):
#        plt.annotate(pat_num[i][j]+1,(x[j],y[j]))
    legend_elements = [Line2D([0], [0],marker='.', color='w', label='patient data',
                              markerfacecolor='k', markersize=15),
                       Line2D([0], [0], marker='*', color='w', label='cluster centers',
                              markerfacecolor='k', markersize=15)]
#    plt.legend(handles=legend_elements,bbox_to_anchor=(0.75,1.15),fontsize=15)
    plt.tight_layout()
    with PdfPages('cluster_variance_plot.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
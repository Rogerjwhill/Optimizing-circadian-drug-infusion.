"""
Created on Fri Mar 23 13:37:40 2018

@author: roger
"""

import numpy as np

from matplotlib import pyplot as plt
from sklearn import manifold
import Useful_functions as uf
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import  pdist, squareform
from matplotlib.lines import Line2D
import drug_data as dd
import skfuzzy as fuzz



pat_num = [dd.pat_nums_cpt,dd.pat_nums_fu,dd.pat_nums_lohp]
drug_names = ['_cpt11','_5fu','_lohp']
titles = ['CPT11 ','5-FU ','L-OHP ']
clusters_colour =  uf.get_distinct_colors(11)#['r','g','m','y','grey','cyan','orange','brown']
point_style = ['o','s','^','h','>']
point_color = uf.get_distinct_colors(11)#['0','0.5','0.35','0.1','0.7']
metric = ['euclidean']
max_cluster = [10,8,9]

for i in [2]:
    k = np.loadtxt('data/pat_param_drug'+drug_names[i]+'.txt')

    # Set up the loop and plot
    
    fig1, axes1 = plt.subplots(3,3, figsize=(10, 10))
    alldata = np.transpose(k)
    V_sf = []

    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        if ncenters>max_cluster[i]:
            ax.remove()
        else:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                alldata, ncenters, 2, error=1e-13, maxiter=50000, init=None)
    
            # Store fpc values for later
            v_sf = 0
            for m in range(ncenters):
                for l in range(len(k)):
                    v_sf += u[m][l]**2*np.linalg.norm(k[l] -cntr[m]) - u[m][l]**2*np.linalg.norm(cntr[m] - np.mean(cntr,0))
            V_sf.append(v_sf)
    
    
            # Plot assigned clusters, for each data point in training set
            cluster_membership = np.argmax(u, axis=0)
            all_data = np.append(k,cntr,axis=0)
            dists = squareform(pdist(all_data))
            mds = manifold.MDS(n_components=2, dissimilarity='precomputed')
            pos = mds.fit(dists).embedding_
    
            for j in range(len(k)):
                ax.scatter((pos[j, 0]- min(pos[:,0]))/(max(pos[:,0])-min(pos[:,0])), (pos[j, 1]- min(pos[:,1]))/(max(pos[:,1])-min(pos[:,1]))  ,c=clusters_colour[cluster_membership[j]])
            for n in range(len(cntr)):
                ax.plot((pos[j+n+1, 0]- min(pos[:,0]))/(max(pos[:,0])-min(pos[:,0])), (pos[j+n+1, 1]- min(pos[:,1]))/(max(pos[:,1])-min(pos[:,1])),'*',markersize=10,markeredgecolor='k',markerfacecolor=clusters_colour[n])
            plt.title(titles[i])
            plt.tight_layout()
    
            ax.set_title('Centers = {0};\n V_FS = {1:.3f}'.format(ncenters, v_sf))
#        ax.axis('off')

#    legend_elements = [Line2D([0], [0],marker='.', color='w', label='patient data',
#                          markerfacecolor='k', markersize=15),
#                   Line2D([0], [0], marker='*', color='w', label='cluster centers',
#                          markerfacecolor='k', markersize=15)]
#
#
#    ax.legend(handles=legend_elements,bbox_to_anchor=(0.5,1.2),fontsize=15)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(len(V_sf))+2, V_sf)
    ax2.set_xticks(np.arange(len(V_sf))+2)
    ax2.set_xlabel("Number of centers",fontsize=20)
    ax2.set_ylabel("V_FS",fontsize=20)





    folder_path = uf.new_folder('manifold_cluster_plots')
    with PdfPages(folder_path+'manifold_fuzzy_clusters' + drug_names[i] + '.pdf') as pdf:
        pdf.savefig(fig1, bbox_inches='tight')

    with PdfPages(folder_path+'V_sf_plot' + drug_names[i] + '.pdf') as pdf:
            pdf.savefig(fig2, bbox_inches='tight')

    group_num = V_sf.index(np.min(V_sf))+2
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                alldata, group_num , 2, error=0.001, maxiter=20000, init=None)


    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    np.save('data/cluster_membership'+drug_names[i],cluster_membership)
    print(cluster_membership)
    all_data = np.append(k,cntr,axis=0)
    dists = squareform(pdist(all_data))
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed')
    pos = mds.fit(dists).embedding_
    np.save('MDS_points'+drug_names[i],pos)
    fig = plt.figure(figsize=(8,8))
    for j in range(len(k)):
        plt.scatter((pos[j, 0]- min(pos[:,0]))/(max(pos[:,0])-min(pos[:,0])), (pos[j, 1]- min(pos[:,1]))/(max(pos[:,1])-min(pos[:,1])) ,marker=point_style[cluster_membership[j]],color=point_color[cluster_membership[j]])
    for n in range(len(cntr)):
        plt.plot((pos[j+n+1, 0]- min(pos[:,0]))/(max(pos[:,0])-min(pos[:,0])), (pos[j+n+1, 1]- min(pos[:,1]))/(max(pos[:,1])-min(pos[:,1])),'*',markersize=10,markeredgecolor='k',markerfacecolor=point_color[n])

    x = (pos[:, 0]- min(pos[:,0]))/(max(pos[:,0])-min(pos[:,0])) + 0.02
    y = (pos[:, 1]- min(pos[:,1]))/(max(pos[:,1])-min(pos[:,1]))
#    for j, txt in enumerate(pos[0:-group_num]):
#        plt.annotate(pat_num[i][j]+1,(x[j],y[j]))

    legend_elements = [Line2D([0], [0],marker='.', color='w', label='patient data',
                          markerfacecolor='k', markersize=15),
                   Line2D([0], [0], marker='*', color='w', label='cluster centers',
                          markerfacecolor='k', markersize=15)]


    plt.legend(handles=legend_elements,bbox_to_anchor=(0.75,1.15),fontsize=15)
    plt.show()
#    print(group_num,fpc,np.max(fpcs))

    with PdfPages(folder_path+'MDS_cluster_final_plot' + drug_names[i] + '.pdf') as pdf:
            pdf.savefig(fig, bbox_inches='tight')
#
#    fig = plt.figure()
#    a,b=cntr.shape
#    plt.scatter(pdist(all_data),pdist(pos),color='k')
#    plt.xlabel('original distance')
#    plt.ylabel('projected distance')
#    plt.title('correlation = '+str(round(np.corrcoef(pdist(all_data),pdist(pos))[0,1],3)))
#    with PdfPages(folder_path+'sheperd_diagram_corr' + drug_names[i] + '.pdf') as pdf:
#            pdf.savefig(fig, bbox_inches='tight')
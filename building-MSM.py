from msmbuilder.featurizer import AtomPairsFeaturizer
from msmbuilder.cluster import KCenters
import numpy as np
from msmbuilder.decomposition import tICA
from msmbuilder.dataset import dataset
from matplotlib import pyplot as plt
import matplotlib as mp
import os
from msmbuilder.tpt import net_fluxes
import sys
"""
xyz = dataset('../xtc/*.xtc', topology = '~/Desktop/tica-projection/Structures/Reference-PRE.pdb')
list1=np.loadtxt('atompairs-5pairs-5helix-P')
featurizer = AtomPairsFeaturizer(pair_indices=list1)

ticadist = xyz.fit_transform_with(featurizer, 'atompairsfeaturizer/', fmt='dir-npy')

#ticadist =dataset('../atompairsfeaturizer/',mode='r',fmt='dir-npy',verbose=True)
tica_model=tICA(lag_time=400,n_components=2)
tica_model=ticadist.fit_with(tica_model)
tica_trajs = ticadist.transform_with(tica_model, 'tica/',fmt='dir-npy')
"""
tica_trajs=dataset('./tica',mode='r',fmt='dir-npy',verbose=True)
txx = np.concatenate(tica_trajs)
clusterer = KCenters(n_clusters=1000,random_state=8)
#clusterer = dataset('./cluster',mode='r',fmt='dir-npy',verbose=True)
clustered_trajs = tica_trajs.fit_transform_with(clusterer, 'cluster-test/', fmt='dir-npy')

"""
from msmbuilder.msm import MarkovStateModel, implied_timescales
data=dataset('cluster',mode='r',fmt='dir-npy',verbose=True)
lag_times=range(100,1300,100)
msm_timescales = implied_timescales(data, lag_times, n_timescales=10,msm=MarkovStateModel(lag_time=250,reversible_type='transpose',ergodic_cutoff='off'))
np.savetxt('msm_timescales_2.txt',msm_timescales)


#data=np.loadtxt('frame')
#data1=np.loadtxt('frame-2211-2216')
txx = np.concatenate(tica_trajs)
plt.hexbin(txx[:,0], txx[:,1],bins='log', mincnt=0.1, cmap='viridis')
plt.savefig('tica.png')

txx = np.concatenate(tica_trajs)
#plt.hexbin(txx[:,0], txx[:,1],bins='log', mincnt=0.1, cmap='viridis')
#for i in range(200,300):
#	data=np.load('./tica/00000'+str(i)+'.npy')
#	plt.hexbin(txx[:,0], txx[:,1],bins='log', mincnt=0.1, cmap='viridis')
#	plt.plot(data[:,0], data[:,1],color = 'red', linewidth = 0.3)
#	plt.savefig('tica-'+str(i)+'projections.png')
#	plt.close()

#txx = np.concatenate(tica_trajs)
#plt.hexbin(txx[:,0], txx[:,1],bins='log', mincnt=0.1, cmap='viridis')
#for i in range(200,300):
#	data=np.load('./tica/00000'+str(i)+'.npy')
#	plt.hexbin(txx[:,0], txx[:,1],bins='log', mincnt=0.1, cmap='viridis')
#	plt.plot(data[:,0], data[:,1],color = 'red', linewidth = 0.3)
#plt.savefig('tica-100projections.png')

plt.hexbin(txx[:,0], txx[:,1],bins='log', mincnt=0.1, cmap='viridis')
#plt.plot(data[:,0], data[:,1],color = 'red', linewidth = 0.3)
plt.scatter(clusterer.cluster_centers_[:,0],clusterer.cluster_centers_[:,1],s=1, c='w')
plt.savefig('cluster.png')

"""

from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump

msm = MarkovStateModel(lag_time=750, reversible_type='transpose',ergodic_cutoff='off')
msm.fit(clustered_trajs)
#net_fluxes(0,2,msm)
#print("now output MSM lag time is 5ns")
#print(len(msm.state_labels_))
#np.savetxt('TCM.txt',msm.countsmat_)
#np.savetxt('state_lables.txt', msm.state_labels_)
#np.savetxt('TPM.txt', msm.transmat_)
#np.savetxt("population.txt", msm.populations_)

from msmbuilder.lumping import PCCAPlus
pcca = PCCAPlus.from_msm(msm, n_macrostates=4)
macro_trajs = pcca.transform(clustered_trajs)

#plt.hexbin(txx[:, 0], txx[:, 1], bins='log', mincnt=0.1, cmap="Greys")
plt.scatter(clusterer.cluster_centers_[msm.state_labels_, 0],
            clusterer.cluster_centers_[msm.state_labels_, 1],
            s=5,
            c=pcca.microstate_mapping_,
)
#plt.plot(data[:,0], data[:,1],color = 'red', linewidth = 1,marker='o',markersize=3)
#plt.plot(data1[:,0], data1[:,1],color = 'green', linewidth = 1,marker='o',markersize=3)
plt.xlabel('tIC 1')
plt.ylabel('tIC 2')
plt.savefig('4macro.png')

np.savetxt('pcca_4.txt', pcca.microstate_mapping_)


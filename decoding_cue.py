import function_2p.load as load_2p
import os
import pickle
import random
import numpy as np
load_2p.import_functionmodule()
import functions.analysis as analysis
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.svm import LinearSVC

directory = r'D:\OneDrive - UCSF\2p'
mouselist = ['HJ_2p_ext_M2','HJ_2p_ext_F6','HJ_2p_ext_M3']
daylist = [list(range(8,21))+[22],list(range(9,22))+[23],[6,7,8,9,10,11,12,14,15,16,17,18,19,20]] # [3 days of acquisition, 5 days of extinction, 5 days of degradation, 1 day of reacquisition]

cuelabel = 15
rewardlabel = 10
cuerewdelay = 3000
ntrial = 30

nplanes = 3
datatype = 'fnorm'

subsampling = np.nan
if np.isnan(subsampling):
    niteration = 1
else:
    niteration = 100

accuracy = [[] for x in mouselist]
selectivity = [[] for x in mouselist]
pval = [[] for x in mouselist]
for imouse, mouse in enumerate(mouselist):
    accuracy[imouse] = [[] for x in range(len(daylist[imouse]))]
    selectivity[imouse] = [[] for x in range(len(daylist[imouse]))]
    pval[imouse] = [[] for x in range(len(daylist[imouse]))]

    ncell = [np.nan for x in range(len(daylist[imouse]))]
    for iday, day in enumerate(daylist[imouse]):
        full_directory = os.path.join(directory, mouse, 'Day' + str(day), 'data.pkl')
        try:
            with open(full_directory, 'rb') as file:
                pickle_data = pickle.load(file)
            ncell[iday] = int(np.sum(pickle_data['data']['iscell']))
        except:
            ncell[iday] = np.nan
    mincell = np.nanmin(ncell)
    ncellintrain = round(mincell*subsampling)

    for iday, day in enumerate(daylist[imouse]):
        if np.isnan(ncell[iday]):
            continue

        full_directory = os.path.join(directory, mouse, 'Day' + str(day), 'data.pkl')
        with open(full_directory, 'rb') as file:
            pickle_data = pickle.load(file)

        print(day)
        eventlog = pickle_data['eventlog']
        eventindex = eventlog[:, 0]
        eventtime = eventlog[:, 1]
        falseeventflag = eventlog[:, 2]

        data = pickle_data['data']
        time = pickle_data['time']

        cuetime = eventtime[eventindex == cuelabel]
        cuetime = cuetime[:ntrial]
        preauc = []
        postauc = []
        for iplane in range(nplanes):
            data_plane = data[datatype][np.logical_and(data['iscell'] == 1, data['plane'] == iplane)]
            auc_temp = [analysis.calculate_deltaauc(eventindex, eventtime, d, time[iplane], cuetime,
                                                    [-cuerewdelay,0], [0,cuerewdelay],normalized=True)
                        for d in data_plane]
            preauc_temp = np.array([x[1] for x in auc_temp])
            postauc_temp = np.array([x[2] for x in auc_temp])

            preauc.append(preauc_temp)
            postauc.append(postauc_temp)
        preauc = np.concatenate(preauc)
        postauc = np.concatenate(postauc)

        selectivity[imouse][iday] = [(np.mean(x)-np.mean(y))/np.sqrt(np.var(x)+np.var(y)) for x,y in zip(postauc,preauc)]
        pval_temp = [stats.ttest_rel(x,y) for x,y in zip(preauc,postauc)]
        pval[imouse][iday] = [x.pvalue for x in pval_temp]

        performance = []
        for iiter in range(niteration):
            performance_temp = []
            if np.isnan(subsampling):
                initer = range(ncell[iday])
            else:
                initer = random.sample(range(ncell[iday]),ncellintrain)
            for itrial in range(ntrial):
                traindata = np.hstack((preauc[:,[x for x in list(range(ntrial)) if x!=itrial]],
                                      postauc[:,[x for x in list(range(ntrial)) if x!=itrial]])).T
                traindata = traindata[:,initer]
                trainlabel = np.hstack((np.zeros(ntrial-1),np.ones(ntrial-1)))
                testdata = np.vstack((preauc[:,itrial],postauc[:,itrial]))
                testdata = testdata[:,initer]

                clf = LinearSVC().fit(traindata, trainlabel)
                testlabel = clf.predict(testdata)
                performance_temp.append(testlabel==[0,1])
            performance.append(np.mean(np.concatenate(performance_temp)))

        accuracy[imouse][iday] = performance

plt.close('all')
fig = plt.figure(figsize=(4,3))
mean_accuracy = [np.array([np.nan if len(x)==0 else np.mean(xx) for xx in x]) for x in accuracy]
std_accuracy = [np.array([np.nan if len(x)==0 else np.std(xx)/np.sqrt(niteration) for xx in x]) for x in accuracy]
[plt.errorbar(range(len(m)),m,s,color='grey',linewidth=0.5) for m,s in zip(mean_accuracy,std_accuracy)]
plt.plot(np.nanmean(mean_accuracy,axis=0),color='k',linewidth=1)
plt.plot([2.5,2.5,np.nan,7.5,7.5,np.nan,12.5,12.5],[0.5,0.9,np.nan,0.5,0.9,np.nan,0.5,0.9],'k:')
plt.xlabel('Session')
plt.ylabel('Decoding accuracy')
plt.savefig(os.path.join(directory,'decoding_accuracy_nontracking_'+str(subsampling)+'subsampling.pdf'), bbox_inches='tight')

cmap = plt.get_cmap('tab10')
values = np.linspace(0, 1, 4)  # Values from 0 to 1
colors = [cmap(v) for v in values]

plt.close('all')
fig = plt.figure(figsize=(6,8))
plt.subplot2grid((4,1),(0,0))
[plt.plot([np.nan if len(xx)==0 else len(xx) for xx in x],color=y,marker='o') for x,y in zip(pval,colors)]
plt.plot([2.5,2.5,np.nan,7.5,7.5,np.nan,12.5,12.5],[0,900,np.nan,0,900,np.nan,0,900],'k:')
plt.ylabel('# of neurons')

plt.subplot2grid((4,1),(1,0))
[plt.plot([np.nan if len(xx)==0 else sum(np.array(xx)<0.01) for xx in x],linestyle=':',color=y,marker='o') for x,y in zip(pval,colors)]
plt.plot([2.5,2.5,np.nan,7.5,7.5,np.nan,12.5,12.5],[0,150,np.nan,0,150,np.nan,0,150],'k:')
plt.ylabel('# of cue responsive')

plt.subplot2grid((4,1),(2,0))
[plt.plot([np.nan if len(xx)==0 else np.mean(np.array(xx)<0.01) for xx in x],color=y,marker='o') for x,y in zip(pval,colors)]
#[plt.plot([np.nan if len(xx)==0 else np.mean(np.array(xx)<0.01) for xx in x],color=y,linestyle='--') for x,y in zip(pval,colors)]
plt.plot([2.5,2.5,np.nan,7.5,7.5,np.nan,12.5,12.5],[0,0.2,np.nan,0,0.2,np.nan,0,0.2],'k:')
plt.ylabel('fon (p<0.01)')

plt.subplot2grid((4,1),(3,0))
[plt.plot([np.nan if len(xx)==0 else np.mean(np.array(xx)<0.05) for xx in x],color=y,marker='o') for x,y in zip(pval,colors)]
#[plt.plot([np.nan if len(xx)==0 else np.mean(np.array(xx)<0.01) for xx in x],color=y,linestyle='--') for x,y in zip(pval,colors)]
plt.plot([2.5,2.5,np.nan,7.5,7.5,np.nan,12.5,12.5],[0,0.3,np.nan,0,0.3,np.nan,0,0.3],'k:')
plt.ylabel('fon (p<0.05)')
plt.pause(1)
plt.savefig(os.path.join(directory,'fon_nontracking.pdf'), bbox_inches='tight')

ylimit =[40,30,30]
plt.close('all')
fig = plt.figure(figsize=(20,8))
for im in range(len(mouselist)):
    for i,(s,p) in enumerate(zip(selectivity[im],pval[im])):
        s = np.array(s)
        p = np.array(p)
        plt.subplot2grid((len(mouselist),len(selectivity[im])),(im,i))
        plt.hist(s[p<0.05],bins=np.arange(-1,1,0.1))
        plt.ylim([0,ylimit[im]])
plt.savefig(os.path.join(directory,'fon_nontracking.pdf'), bbox_inches='tight')

plt.close('all')
plt.figure()
for im in range(len(mouselist)):
    absselec = [np.nan if len(s)==0 else [abs(ss) for ss,pp in zip(s,p) if pp<0.05] for s,p in zip(selectivity[im],pval[im])]
    plt.errorbar(range(len(absselec)),[np.nan if np.isnan(x).any() else np.median(x) for x in absselec],
                 [np.nan if np.isnan(x).any() else np.std(x)/np.sqrt(len(x)) for x in absselec],color=colors[im])
plt.plot([2.5,2.5,np.nan,7.5,7.5,np.nan,12.5,12.5],[0.1,0.4,np.nan,0.1,0.4,np.nan,0.1,0.4],'k:')
plt.savefig(os.path.join(directory,'aveselec_nontracking.pdf'), bbox_inches='tight')








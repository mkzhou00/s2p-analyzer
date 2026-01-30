# common functions for all opto experiments

from sklearn.metrics import roc_auc_score as auROC
from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats
import statsmodels.api as sm
import copy
import scipy.io as sio
import os
import pandas as pd

def argsort_forlist(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)

def central_tendency(x, y, measure='mean_difference'):
    x = x.astype(float)
    y = y.astype(float)
    if measure=='mean_difference':
        return np.mean(x)-np.mean(y)
    elif measure=='mean_x':
        return np.mean(x)
    elif measure=='auROC':
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        data = np.concatenate((x, y))
        labels = np.concatenate((np.ones(x.size,), np.zeros(y.size,)))
        return 2*auROC(labels, data)-1
    elif measure=='varbymean':        
        return np.var(x-y)/np.mean(x-y)
    
def CDFplot(x, ax, color=None, label='', linetype='-'):
    x = np.array(x)
    ix=np.argsort(x)
    ax.plot(x[ix], ECDF(x)(x)[ix], linetype, color=color, label=label)
    return ax

def Benjamini_Hochberg_pvalcorrection(vector_of_pvals):
    # This function implements the BH FDR correction
    # Parameters:
    # Vector of p values from the different tests
    # Returns: Corrected p values.
    
    sortedpvals = np.sort(vector_of_pvals)
    orderofpvals = np.argsort(vector_of_pvals)
    m = sortedpvals[np.isfinite(sortedpvals)].size # Total number of hypotheses
    corrected_sortedpvals = np.nan*np.ones((sortedpvals.size,))
    corrected_sortedpvals[m-1] = sortedpvals[m-1]
    for i in range(m-2, -1, -1):
        corrected_sortedpvals[i] = np.amin([corrected_sortedpvals[i+1], sortedpvals[i]*m/(i+1)])
    correctedpvals = np.nan*np.ones((vector_of_pvals.size,))
    correctedpvals[orderofpvals] = corrected_sortedpvals
    return correctedpvals

def standardize_plot_graphics(ax):
    [i.set_linewidth(0.5) for i in ax.spines.values()]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

def t_test(x,y,alternative='two-sided'):
    tvalue, double_p = stats.ttest_ind(x,y,equal_var = False)
    if alternative == 'two-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval

def convert_pvalue_to_asterisks(pvalues):
    psymbols = copy.deepcopy(pvalues)
    for condition in pvalues.keys():
        for cue in pvalues.keys():
            if float(pvalues[cue]) <= 0.0001:
                psymbols[cue] = "****"
            elif float(pvalues[cue]) <= 0.001:
                psymbols[cue] = "***"       
            elif float(pvalues[cue]) <= 0.01:
                psymbols[cue] = "**"
            elif float(pvalues[cue]) <= 0.05:
                psymbols[cue] = "*"
            else:
                psymbols[cue] = 'ns'
    return psymbols
   
def label_diff(i,j,text,X,Y,ax):
    x = (X[i]+X[j])/2
    y = 1.5*max(Y[i], Y[j])
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':20,'shrinkB':20,'linewidth':2}
    ax.annotate(text, xy=(X[i],y+7), zorder=10)
    ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)

def point_to_line(pt, p1, p2):
    d = np.empty([len(pt),1])
    d.fill(np.nan)
    
    line = p2 - p1
    for i in range(len(pt)):
        b = p1 - pt[i,:]
        d[i] = np.linalg.norm(np.cross(line, b))/np.linalg.norm(line)
    return d

def getCumsumChangePoint(trials_list_x, cumsum_data_y, percent_max_dist = 0.95, data_direction = 'increase',  animal_name ='',):
    '''
    

    Parameters
    ----------
    trials_list_x : numpy array
        DESCRIPTION.
    cumsum_data_y : numpy array
        DESCRIPTION.
    percent_max_dist : float between 0 and 1, optional
        DESCRIPTION. The default is 0.8.
    data_direction : TYPE, optional
        DESCRIPTION. The default is 'increase'.

    Returns
    -------
    learned_trial_params : dict
        DESCRIPTION.

    '''
        
    
    # full_trials_x = trials_list_x
    # full_trials_y = cumsum_data_y

        
    diagonal_y = np.linspace(0, cumsum_data_y[-1], len(trials_list_x)) #draw diagonal from 0 to end of y that's length of data/trials

    cumsum_data_all = np.stack((trials_list_x, cumsum_data_y), axis=1).reshape(-1, 2)
    dist_from_diag = np.ravel(point_to_line(cumsum_data_all, cumsum_data_all[0,:], cumsum_data_all[-1,:])) # calc distance between diagonal and datapoints
    
        
    max_dist = np.max(dist_from_diag) 
    maxdist_cutoff = trials_list_x[np.flatnonzero(dist_from_diag>=(max_dist*percent_max_dist))] #find the first point that is within 75% of max distance from diagonal
    learned_trial = maxdist_cutoff[0]
    diagonal_x = trials_list_x 
    
    
    if data_direction == 'increase':
        # if diagonal is under data at 'learned trial' (ie change trial captured is for decreasing data)
        # then iterate backwards over trials and data applying same algorithm until diagonal is above data at learned trial
        if (cumsum_data_y[np.flatnonzero(trials_list_x==learned_trial)][0]) > (diagonal_y[np.flatnonzero(trials_list_x == learned_trial)][0]):
        
            for last_x, last_y in zip (np.flip(trials_list_x[trials_list_x <= learned_trial]), np.flip(cumsum_data_y[trials_list_x <= learned_trial])): 
                diagonal_y = np.linspace(0, last_y, len(trials_list_x[trials_list_x <= last_x])) 
        
                cumsum_data_all = np.stack((trials_list_x[trials_list_x <= last_x], cumsum_data_y[trials_list_x <= last_x]), axis=1).reshape(-1, 2)
                dist_from_diag = np.ravel(point_to_line(cumsum_data_all, cumsum_data_all[0,:], cumsum_data_all[-1,:]))
                max_dist = np.max(dist_from_diag)
                maxdist_cutoff = trials_list_x[:last_x][np.flatnonzero(dist_from_diag >= (max_dist*percent_max_dist))]
                learned_trial = maxdist_cutoff[0]
                diagonal_x = trials_list_x[trials_list_x <= last_x]
                if (cumsum_data_y[np.flatnonzero(trials_list_x == learned_trial)][0]) < (diagonal_y[np.flatnonzero(trials_list_x == learned_trial)][0]):
                    break
            else: #in case loop never breaks out
                raise ValueError('no inflection point found for '+animal_name +' on correct side of diagonal')
   
    elif data_direction == 'decrease': 
        if (cumsum_data_y[np.flatnonzero(trials_list_x==learned_trial)][0]) < (diagonal_y[np.flatnonzero(trials_list_x == learned_trial)][0]):
        
            for last_x, last_y in zip (np.flip(trials_list_x[trials_list_x <= learned_trial]), np.flip(cumsum_data_y[trials_list_x <= learned_trial])): 
                diagonal_y = np.linspace(0, last_y, len(trials_list_x[trials_list_x <= last_x])) 
        
                cumsum_data_all = np.stack((trials_list_x[trials_list_x <= last_x], cumsum_data_y[trials_list_x <= last_x]), axis=1).reshape(-1, 2)
                dist_from_diag = np.ravel(point_to_line(cumsum_data_all, cumsum_data_all[0,:], cumsum_data_all[-1,:]))
                max_dist = np.max(dist_from_diag)
                maxdist_cutoff = trials_list_x[:last_x][np.flatnonzero(dist_from_diag >= (max_dist*percent_max_dist))]
                learned_trial = maxdist_cutoff[0]
                diagonal_x = trials_list_x[trials_list_x <= last_x]
                if (cumsum_data_y[np.flatnonzero(trials_list_x == learned_trial)][0]) > (diagonal_y[np.flatnonzero(trials_list_x == learned_trial)][0]):
                    break
            else: #in case loop never breaks out
                raise ValueError('no inflection point found for ' +animal_name +'on correct side of diagonal')
    
    #recalculate max distance in cordinates where y is normalized to 1
    cumsum_data_normed = cumsum_data_all
    cumsum_data_normed[:,1] = cumsum_data_normed[:,1]/ cumsum_data_normed[-1,1]
    dist_from_diag_norm = np.ravel(point_to_line(cumsum_data_normed, cumsum_data_normed[0,:], cumsum_data_normed[-1,:])) # calc distance between diagonal and datapoints
    max_dist_norm = np.max(dist_from_diag_norm) 
   

    
    learned_trial_params = {'learned_trial': learned_trial, 'diag_y': diagonal_y, 'diag_x': diagonal_x, 'dist_from_diag': dist_from_diag, 'max_dist': max_dist, 
                            'dist_at_learned_trial': max_dist*percent_max_dist, 'max_dist_norm':max_dist_norm, 'dist_at_learned_trial_norm': max_dist_norm*percent_max_dist  }
    return learned_trial_params

def plot_raw_licks(CS, licks, before, after, ax=None, cue_color=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    for i in range(0, CS.shape[0]):
        raw_licks = licks['Timestamp'].loc[(licks['Timestamp'] >= ((CS['Timestamp'].iloc[i])+before)) & (licks['Timestamp'] <= (CS['Timestamp'].iloc[i] + after))]
        ax.vlines((raw_licks/1000-CS['Timestamp'].iloc[i]/1000), i, i+1, linewidth=1, color='#654321')
        # ax.vlines(0, i, i+1, linewidth=1, linestyles='--', color='k')
    return ax


def get_changepoint_params(data, cue_types, threshold:float):
    abruptness = {}    
    learnedtrial = {}
    meanslope = {}
    for ct, cue_type in enumerate(cue_types):
        abruptness[cue_type] = {}
        learnedtrial[cue_type] = {}
        meanslope[cue_type] = {}

        animals = list(set(data.keys())) #get animals in the condition
        for a, animal in enumerate(animals): 
            if cue_type != 'CS3':
                y = data[animal][cue_type]
                x = np.arange(0, (len(y)), 1) #generate x axis
                learned_trial_params = getCumsumChangePoint(x, y, animal_name=animal, percent_max_dist=threshold)
                tempabruptness = learned_trial_params['dist_at_learned_trial_norm']
                templearnedtrial = learned_trial_params['learned_trial']
                tempmeanslope = float(y[-1] - y[templearnedtrial]) / (len(y) - templearnedtrial)   

                if tempmeanslope > 0.5:
                    abruptness[cue_type][animal] = tempabruptness
                    learnedtrial[cue_type][animal] = templearnedtrial
                    meanslope[cue_type][animal] = tempmeanslope
                else:
                    abruptness[cue_type][animal] = np.nan
                    meanslope[cue_type][animal] = np.nan     
                    learnedtrial[cue_type][animal] = np.nan
                    # if len(y) >= 295 and len(y) < 400 :
                    #     learnedtrial[cue_type][animal] = len(y)
                    # else:
                    #     learnedtrial[cue_type][animal] = 400
            else: 
                y = data[animal][cue_type]
                tempmeanslope = float(y[-1] - y[-100]) / 100
                if tempmeanslope > 0.5:
                    meanslope[cue_type][animal] = tempmeanslope
                else:
                    meanslope[cue_type][animal] = np.nan
    return abruptness, learnedtrial, meanslope


def extract_licks_per_trial(CS, lick, nlicksCS, nlicksbaseCS, onset_from_cue, offset_from_cue):
    for ics in range(0, CS.shape[0]):
        tempanticipatorylicksCS = lick["Timestamp"].loc[
            (lick["Timestamp"] >= CS["Timestamp"].iloc[ics] + onset_from_cue)
            & (lick["Timestamp"] < (CS["Timestamp"].iloc[ics] + offset_from_cue))
        ]
        nlicksCS[ics, 0] = len(tempanticipatorylicksCS)
        duration = offset_from_cue - onset_from_cue
        baselinelicksCS = lick["Timestamp"].loc[
            (lick["Timestamp"] < CS["Timestamp"].iloc[ics])
            & (lick["Timestamp"] >= (CS["Timestamp"].iloc[ics] - duration))
        ]
        nlicksbaseCS[ics, 0] = len(baselinelicksCS)


def get_df(animal, day, cue, nlicksCS, nlicksbaseCS):
    DF_COLUMMS = ["Animal", "Day", "Cue", "nlicks fullcue", "nlicks baseline"]

    data = np.column_stack(
        [
            [animal] * nlicksCS.shape[0],
            [day] * nlicksCS.shape[0],
            [cue] * nlicksCS.shape[0],
            nlicksCS,
            nlicksbaseCS,
        ]
    )
    df = pd.DataFrame(data=data, columns=DF_COLUMMS)
    return df


def extract_lick_psth_per_trial(cue_df, lick_df, pre_ms, post_ms, bin_ms):
    """
    Returns psth_counts: (n_trials, n_bins) counts per bin (NOT Hz yet)
            t_centers_s: (n_bins,) time centers in seconds
    Assumes cue_df['Timestamp'] and lick_df['Timestamp'] are in ms.
    """
    cue_times = cue_df["Timestamp"].to_numpy().astype(float)
    lick_times = lick_df["Timestamp"].to_numpy().astype(float)

    edges = np.arange(pre_ms, post_ms + bin_ms, bin_ms, dtype=float)
    t_centers_s = (edges[:-1] + edges[1:]) / 2 / 1000.0
    n_bins = len(edges) - 1

    psth = np.zeros((len(cue_times), n_bins), dtype=float)

    for i, ct in enumerate(cue_times):
        rel = lick_times - ct
        rel = rel[(rel >= pre_ms) & (rel < post_ms)]
        if rel.size:
            psth[i], _ = np.histogram(rel, bins=edges)

    return psth, t_centers_s


def produce_df_for_all_data(indir, onset_from_cue=0, offset_from_cue=3000, pre_ms=-3000, post_ms=6000, bin_ms=100):
    
    # Initialize alldata df and PSTH data for plotting
    DF_COLUMMS = ["Animal", "Day", "Cue", "nlicks fullcue", "nlicks baseline"]

    alldata = pd.DataFrame(columns=DF_COLUMMS)
    psth_rows = []
    
    tempanimals = next(os.walk(indir))[1]
    animals = [a for a in tempanimals if "MZ" in a]
    for animal in animals:
        print(animal)
        datadir = os.path.join(indir, animal)
        # print(datadir)
        tempmatfiles = next(os.walk(datadir))[2]
        matfiles = [
            f
            for f in tempmatfiles
            if "results" not in f and os.path.splitext(f)[1] == ".mat"
        ]

        day_nums = (
            {}
        )  # Run number for each experiment within an animal. 3 runs per animal in the design

        for matfile in matfiles:
            timestamps = [
                matfile.split("_")[-1].split(".")[0] for matfile in matfiles
            ]
            temp = np.argsort(timestamps)
            tempidx = list(temp)
            for i in range(len(temp)):
                tempidx[temp[i]] = i
            for t, timestamp in enumerate(timestamps):
                day_nums[timestamp] = tempidx[t] + 1

        for matfile in matfiles:
            # print(matfile)
            timeofsession = matfile.split("_")[-1].split(".")[0]
            day = day_nums[timeofsession]
            temp = os.path.join(datadir, matfile)
            #             print matfile, timeofsession, day
            behaviordata = sio.loadmat(
                os.path.join(datadir, os.path.splitext(matfile)[0])
            )
            eventdata = behaviordata["eventlog"]
            eventdf = pd.DataFrame(
                data=eventdata, columns=["Events", "Timestamp", "Reward"]
            )
            params = behaviordata["params"]
            #             print(params)

            # # For CA1 animals, comment (77-88) if doing DG
            CS1 = eventdf.loc[eventdf["Events"] == 15]
            CS2 = eventdf.loc[eventdf["Events"] == 16]
            CS3 = eventdf.loc[eventdf["Events"] == 17]

            rewards = eventdf.loc[eventdf["Events"] == 10]
            lick3s = eventdf.loc[eventdf["Events"] == 5]

            nlicksCS1 = np.empty([CS1.shape[0], 1])
            nlicksCS2 = np.empty([CS2.shape[0], 1])
            nlicksCS3 = np.empty([CS3.shape[0], 1])
            nlicksbaseCS1 = np.empty([CS1.shape[0], 1])
            nlicksbaseCS2 = np.empty([CS2.shape[0], 1])
            nlicksbaseCS3 = np.empty([CS3.shape[0], 1])

            extract_licks_per_trial(CS1, lick3s, nlicksCS1, nlicksbaseCS1, onset_from_cue, offset_from_cue)
            extract_licks_per_trial(CS2, lick3s, nlicksCS2, nlicksbaseCS2, onset_from_cue, offset_from_cue)
            extract_licks_per_trial(CS3, lick3s, nlicksCS3, nlicksbaseCS3, onset_from_cue, offset_from_cue)

            alldata = pd.concat([alldata,
                get_df(
                    animal,
                    day,
                    "CS1",
                    nlicksCS1,
                    nlicksbaseCS1,
                )],
                ignore_index=True,
            )
            alldata = pd.concat([alldata,
                get_df(
                    animal,
                    day,
                    "CS2",
                    nlicksCS2,
                    nlicksbaseCS2,
                )],
                ignore_index=True,
            )
            alldata = pd.concat([alldata,
                get_df(
                    animal,
                    day,
                    "CS3",
                    nlicksCS3,
                    nlicksbaseCS3,
                )],
                ignore_index=True,
            )
            
            # ---- NEW: PSTH extraction (trial x bin) ----
            for cue_name, cue_df in [("CS1", CS1), ("CS2", CS2), ("CS3", CS3)]:
                psth_counts, t_centers_s = extract_lick_psth_per_trial(
                    cue_df, lick3s, pre_ms=pre_ms, post_ms=post_ms, bin_ms=bin_ms
                )
                # store as long-form rows
                # (trial index local to this session/cue is fine if you only average)
                for tr in range(psth_counts.shape[0]):
                    for b in range(psth_counts.shape[1]):
                        psth_rows.append(
                            {
                                "Animal": animal,
                                "Day": int(day),
                                "Cue": cue_name,
                                "Trial": tr,
                                "Time_s": float(t_centers_s[b]),
                                "Count": float(psth_counts[tr, b]),
                                "Bin_ms": int(bin_ms),
                            }
                        )
            
    alldata['Day'] = alldata['Day'].astype(int)
    psth_df = pd.DataFrame(psth_rows)

    return alldata, psth_df


def produce_data_per_session(alldata, numdays):
    COL_NAME=['Animal','Day','Cue', 'Performance_to_baseline', 'Performance_to_CSminus']
    data_per_session = pd.DataFrame(columns=COL_NAME)
    cue_types = ['CS1', 'CS2', 'CS3']
    numtrials = [25, 25, 50]

    animals = list(set(alldata['Animal']))
    animals_to_remove = []#['OFC-VTAinact_16']#['mOFCinact_32']
    animals = [a for a in animals if a not in animals_to_remove]

    for animal in animals:
        if numdays == None:
            numdays = len(alldata[(alldata['Animal']==animal) & (alldata['Cue']=='CS1')]) / (numtrials[0])
            numdays = int(numdays)
#         print(numdays)
        # mean_licks_per_animal = np.nan*np.ones((numdays, len(cue_types)))
        perf_to_base = np.nan*np.ones((numdays, len(cue_types)))
        perf_to_CSm = np.nan*np.ones((numdays, len(cue_types)))
        for day in range(numdays):
            tempCS1 = np.array(alldata[(alldata['Cue']=='CS1') & (alldata['Day']==int(day+1)) & (alldata['Animal']==animal)]['nlicks fullcue'], dtype=float)
            tempCS1baseline = np.array(alldata[(alldata['Cue']=='CS1') & (alldata['Day']==int(day+1)) & (alldata['Animal']==animal)]['nlicks baseline'], dtype=float)
            tempCS2 = np.array(alldata[(alldata['Cue']=='CS2') & (alldata['Day']==int(day+1)) & (alldata['Animal']==animal)]['nlicks fullcue'], dtype=float)
            tempCS2baseline = np.array(alldata[(alldata['Cue']=='CS2') & (alldata['Day']==int(day+1)) & (alldata['Animal']==animal)]['nlicks baseline'], dtype=float)
            tempCS3 = np.array(alldata[(alldata['Cue']=='CS3') & (alldata['Day']==int(day+1)) & (alldata['Animal']==animal)]['nlicks fullcue'], dtype=float)
            tempCS3baseline = np.array(alldata[(alldata['Cue']=='CS3') & (alldata['Day']==int(day+1)) & (alldata['Animal']==animal)]['nlicks baseline'], dtype=float)
            
            if tempCS1.size>0:
                perf_to_base[day, 0] = central_tendency(tempCS1, tempCS1baseline)
                perf_to_CSm[day, 0] = central_tendency(tempCS1, tempCS3)
            if tempCS2.size>0:
                perf_to_base[day, 1] = central_tendency(tempCS2, tempCS2baseline)
                perf_to_CSm[day, 1] = central_tendency(tempCS2, tempCS3)
            if tempCS3.size>0:
                perf_to_base[day, 2] = central_tendency(tempCS3, tempCS3baseline)

            for ct, cue_type in enumerate(cue_types):
                data = np.column_stack([ 
                                        animal,
                                        int(day+1),
                                        cue_type,
                                        perf_to_base[day, ct],
                                        perf_to_CSm[day, ct],
                                        ])
                data_per_session = pd.concat([data_per_session, pd.DataFrame(data=data,
                                                                            columns=COL_NAME)],
                                                        ignore_index=True)
    return data_per_session


def accumulative_lick_trial_by_trial(alldata, cue_types, numtrials, animals_to_remove=[], mindays=8):
    cumlick_data = {}
    numdays=None
    animals = list(set(alldata['Animal'])) #get animals in the condition
    # animals_to_remove = []#['OFC-VTAinact_16']#['mOFCinact_32']
    animals = [a for a in animals if a not in animals_to_remove]
    for a, animal in enumerate(animals): #for each animal
#             print(animal)
        cumlick_data[animal] = {}
        for ct, cue_type in enumerate(cue_types): #for each cue type
            if numdays == None:
                numdays = len(alldata[(alldata['Animal']==animal) & (alldata['Cue']==cue_type)]) / (numtrials[ct])
                numdays = int(numdays)
#                 print(numdays)
            all_correct_licks = np.array([])
            tempaccum = np.array([])
            for day in range(numdays): #for each day
                #get temp cue licking for this cue
                tempcue = np.array(alldata[(alldata['Cue']==cue_type) & (alldata['Day']==(day+1)) & (alldata['Animal']==animal)]['nlicks fullcue'], dtype=float)
                # get temp baseline licking for this cue
                tempbaseline = np.array(alldata[(alldata['Cue']==cue_type) & (alldata['Day']==(day+1)) & (alldata['Animal']==animal)]['nlicks baseline'], dtype=float)
                #get the corrected licking by subtracting baseline lick from cue licks 
                tempcorrectlick = np.array([tempcue - tempbaseline])
                all_correct_licks = np.append(all_correct_licks, tempcorrectlick)
#                     all_correct_licks = np.append(all_correct_licks, tempcue)
            tempaccum = np.cumsum(all_correct_licks) #get acummulative sum for this cue type for this animal for all days
#                 print(tempaccum)
            cumlick_data[animal][cue_type] = tempaccum
        numdays=None
    return cumlick_data


def get_df_ind(day, cue, nlicksCS, nlicksbaseCS):
    
    DF_COLUMMS = ["Day", "Cue", "nlicks fullcue", "nlicks baseline"]
    data = np.column_stack(
        [
            [day] * nlicksCS.shape[0],
            [cue] * nlicksCS.shape[0],
            nlicksCS,
            nlicksbaseCS,
        ]
    )
    df = pd.DataFrame(data=data, columns=DF_COLUMMS)
    return df

def produce_data_df_ind(datadir, onset_from_cue=0, offset_from_cue=3000):
    
    DF_COLUMMS = ["Day", "Cue", "nlicks fullcue", "nlicks baseline"]
    alldata = pd.DataFrame(columns=DF_COLUMMS)
    
    tempmatfiles = next(os.walk(datadir))[2]
    matfiles = [
        f
        for f in tempmatfiles
        if "results" not in f and os.path.splitext(f)[1] == ".mat"
    ]

    day_nums = (
        {}
    )  # Run number for each experiment within an animal. 3 runs per animal in the design

    for matfile in matfiles:
        timestamps = [
            matfile.split("_")[-1].split(".")[0] for matfile in matfiles
        ]
        temp = np.argsort(timestamps)
        tempidx = list(temp)
        for i in range(len(temp)):
            tempidx[temp[i]] = i
        for t, timestamp in enumerate(timestamps):
            day_nums[timestamp] = tempidx[t] + 1

    for matfile in matfiles:
        # print(matfile)
        timeofsession = matfile.split("_")[-1].split(".")[0]
        day = day_nums[timeofsession]
        temp = os.path.join(datadir, matfile)
        
        behaviordata = sio.loadmat(
            os.path.join(datadir, os.path.splitext(matfile)[0])
        )                
        eventdata = behaviordata["eventlog"]
        eventdf = pd.DataFrame(
            data=eventdata, columns=["Events", "Timestamp", "Reward"]
        )
        params = behaviordata["params"]

        CS1 = eventdf.loc[eventdf["Events"] == 15]
        CS2 = eventdf.loc[eventdf["Events"] == 16]
        CS3 = eventdf.loc[eventdf["Events"] == 17]

        rewards = eventdf.loc[eventdf["Events"] == 10]
        lick3s = eventdf.loc[eventdf["Events"] == 5]

        nlicksCS1 = np.empty([CS1.shape[0], 1])
        nlicksCS2 = np.empty([CS2.shape[0], 1])
        nlicksCS3 = np.empty([CS3.shape[0], 1])
        nlicksbaseCS1 = np.empty([CS1.shape[0], 1])
        nlicksbaseCS2 = np.empty([CS2.shape[0], 1])
        nlicksbaseCS3 = np.empty([CS3.shape[0], 1])

        extract_licks_per_trial(CS1, lick3s, nlicksCS1, nlicksbaseCS1, onset_from_cue, offset_from_cue)
        extract_licks_per_trial(CS2, lick3s, nlicksCS2, nlicksbaseCS2, onset_from_cue, offset_from_cue)
        extract_licks_per_trial(CS3, lick3s, nlicksCS3, nlicksbaseCS3, onset_from_cue, offset_from_cue)

        alldata = pd.concat([alldata,
            get_df_ind(
                day,
                "CS1",
                nlicksCS1,
                nlicksbaseCS1,
            )],
            ignore_index=True,
        )
        alldata = pd.concat([alldata,
            get_df_ind(
                day,
                "CS2",
                nlicksCS2,
                nlicksbaseCS2,
            )],
            ignore_index=True,
        )
        alldata = pd.concat([alldata,
            get_df_ind(
                day,
                "CS3",
                nlicksCS3,
                nlicksbaseCS3,
            )],
            ignore_index=True,
        )
    alldata['Day'] = alldata['Day'].astype(int)            
    return alldata


def produce_data_per_session_ind(alldata, numdays):
    COL_NAME=['Day','Cue', 'Performance_to_baseline', 'Performance_to_CSminus']
    data_per_session = pd.DataFrame(columns=COL_NAME)
    cue_types = ['CS1', 'CS2', 'CS3']
    numtrials = [25, 25, 50]

    if numdays == None:
        numdays = len(alldata[(alldata['Cue']=='CS1')]) / (numtrials[0])
        numdays = int(numdays)
#         print(numdays)
    # mean_licks_per_animal = np.nan*np.ones((numdays, len(cue_types)))
    perf_to_base = np.nan*np.ones((numdays, len(cue_types)))
    perf_to_CSm = np.nan*np.ones((numdays, len(cue_types)))
    for day in range(numdays):
        tempCS1 = np.array(alldata[(alldata['Cue']=='CS1') & (alldata['Day']==int(day+1)) ]['nlicks fullcue'], dtype=float)
        tempCS1baseline = np.array(alldata[(alldata['Cue']=='CS1') & (alldata['Day']==int(day+1))]['nlicks baseline'], dtype=float)
        tempCS2 = np.array(alldata[(alldata['Cue']=='CS2') & (alldata['Day']==int(day+1))]['nlicks fullcue'], dtype=float)
        tempCS2baseline = np.array(alldata[(alldata['Cue']=='CS2') & (alldata['Day']==int(day+1))]['nlicks baseline'], dtype=float)
        tempCS3 = np.array(alldata[(alldata['Cue']=='CS3') & (alldata['Day']==int(day+1))]['nlicks fullcue'], dtype=float)
        tempCS3baseline = np.array(alldata[(alldata['Cue']=='CS3') & (alldata['Day']==int(day+1))]['nlicks baseline'], dtype=float)
        
        if tempCS1.size>0:
            perf_to_base[day, 0] = central_tendency(tempCS1, tempCS1baseline)
            perf_to_CSm[day, 0] = central_tendency(tempCS1, tempCS3)
        if tempCS2.size>0:
            perf_to_base[day, 1] = central_tendency(tempCS2, tempCS2baseline)
            perf_to_CSm[day, 1] = central_tendency(tempCS2, tempCS3)
        if tempCS3.size>0:
            perf_to_base[day, 2] = central_tendency(tempCS3, tempCS3baseline)

        for ct, cue_type in enumerate(cue_types):
            data = np.column_stack([
                                    int(day+1),
                                    cue_type,
                                    perf_to_base[day, ct],
                                    perf_to_CSm[day, ct],
                                    ])
            data_per_session = pd.concat([data_per_session, pd.DataFrame(data=data,
                                                                        columns=COL_NAME)],
                                                    ignore_index=True)
    return data_per_session

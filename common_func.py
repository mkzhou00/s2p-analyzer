# common functions for all opto experiments

from sklearn.metrics import roc_auc_score as auROC
from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats
import statsmodels.api as sm
import copy

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
        for cue in pvalues[condition].keys():
            if float(pvalues[condition][cue]) <= 0.0001:
                psymbols[condition][cue] = "****"
            elif float(pvalues[condition][cue]) <= 0.001:
                psymbols[condition][cue] = "***"       
            elif float(pvalues[condition][cue]) <= 0.01:
                psymbols[condition][cue] = "**"
            elif float(pvalues[condition][cue]) <= 0.05:
                psymbols[condition][cue] = "*"
            else:
                psymbols[condition][cue] = 'ns'
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


def get_stats_learnedpoint(data, conditions, dist='normal', side='two-sided'):
    if len(conditions) > 2:
        pvals = {"e1_e2": {}, "e1_c": {}, "e2_c": {}}
        CS1_c = np.array(list(data['CS1']['Control'].values()))
        CS1_e1 = np.array(list(data['CS1']['Experimental group1'].values()))
        CS1_e2 = np.array(list(data['CS1']['Experimental group2'].values()))

        CS2_c = np.array(list(data['CS2']['Control'].values()))
        CS2_e1 = np.array(list(data['CS2']['Experimental group1'].values()))
        CS2_e2 = np.array(list(data['CS2']['Experimental group2'].values()))
        pvals['e1_e2']['CS1'] = t_test(np.log(CS1_e1), np.log(CS1_e2))
        pvals['e1_c']['CS1']= t_test(np.log(CS1_e1), np.log(CS1_c))
        pvals['e2_c']['CS1']= t_test(np.log(CS1_e2), np.log(CS1_c))
        pvals['e1_e2']['CS2'] = t_test(np.log(CS2_e1), np.log(CS2_e2))
        pvals['e1_c']['CS2'] = t_test(np.log(CS2_e1), np.log(CS2_c))
        pvals['e2_c']['CS2'] = t_test(np.log(CS2_e2), np.log(CS2_c))
    else: 
        pvals = {"e_c":{}}
        CS1_c = np.array(list(data['CS1']['Control'].values()))
        CS1_e = np.array(list(data['CS1']['Experimental group'].values()))
        CS2_c = np.array(list(data['CS2']['Control'].values()))
        CS2_e = np.array(list(data['CS2']['Experimental group'].values()))

        CS1_c = CS1_c[~np.isnan(CS1_c)]
        CS1_e = CS1_e[~np.isnan(CS1_e)]
        CS2_c = CS2_c[~np.isnan(CS2_c)]
        CS2_e = CS2_e[~np.isnan(CS2_e)]

        if dist == 'normal':
            pvals['e_c']['CS1']= t_test(np.log(CS1_e), np.log(CS1_c), alternative=side)
            pvals['e_c']['CS2'] = t_test(np.log(CS2_e), np.log(CS2_c), alternative=side)
        else:
            u_stat, pvals['e_c']['CS1'] = mannwhitneyu(np.log(CS1_e), np.log(CS1_c), alternative=side)
            u_stat, pvals['e_c']['CS2'] = mannwhitneyu(np.log(CS2_e), np.log(CS2_c), alternative=side)

    return pvals


def GLM(data):
    data.dropna(inplace=True)
    model = sm.MixedLM.from_formula("Performance_to_baseline ~ C(Condition, Treatment(reference='Control'))*Day",
                                    data,
                                    groups=data['Animal'])
    result = model.fit(reml=False)    

    model_nointeract = sm.MixedLM.from_formula("Performance_to_baseline ~ Day",
                                                data,
                                                groups=data["Animal"])
    result_nointeract = model_nointeract.fit(reml=False)
    gof = 2*(result.llf - result_nointeract.llf)
    delta_df = result_nointeract.df_resid-result.df_resid
    pval = 1 - stats.chi2.cdf(gof, delta_df)
    return(gof, pval)


def get_changepoint_params(data, conditions, cue_types, threshold:float):
    abruptness = {}    
    learnedtrial = {}
    meanslope = {}
    for ct, cue_type in enumerate(cue_types):
        abruptness[cue_type] = {}
        learnedtrial[cue_type] = {}
        meanslope[cue_type] = {}
        for c, condition in enumerate(conditions):
            abruptness[cue_type][condition] = {}
            learnedtrial[cue_type][condition] = {}
            meanslope[cue_type][condition] = {}     
            animals = list(set(data[condition].keys())) #get animals in the condition
            for a, animal in enumerate(animals): 
                if cue_type != 'CS3':
                    y = data[condition][animal][cue_type]
                    x = np.arange(0, (len(y)), 1) #generate x axis
                    learned_trial_params = getCumsumChangePoint(x, y, animal_name=animal, percent_max_dist=threshold)
                    tempabruptness = learned_trial_params['dist_at_learned_trial_norm']
                    templearnedtrial = learned_trial_params['learned_trial']
                    tempmeanslope = float(y[-1] - y[templearnedtrial]) / (len(y) - templearnedtrial)   

                    if tempmeanslope > 0.5:
                        abruptness[cue_type][condition][animal] = tempabruptness
                        learnedtrial[cue_type][condition][animal] = templearnedtrial
                        meanslope[cue_type][condition][animal] = tempmeanslope
                    else:
                        abruptness[cue_type][condition][animal] = np.nan
                        meanslope[cue_type][condition][animal] = np.nan     
                        learnedtrial[cue_type][condition][animal] = np.nan
                        # if len(y) >= 295 and len(y) < 400 :
                        #     learnedtrial[cue_type][condition][animal] = len(y)
                        # else:
                        #     learnedtrial[cue_type][condition][animal] = 400
                else: 
                    y = data[condition][animal][cue_type]
                    tempmeanslope = float(y[-1] - y[-100]) / 100
                    if tempmeanslope > 0.5:
                        meanslope[cue_type][condition][animal] = tempmeanslope
                    else:
                        meanslope[cue_type][condition][animal] = np.nan
    return abruptness, learnedtrial, meanslope

def mixdedmdl(data):
    data.dropna(inplace=True)
    model = sm.MixedLM.from_formula("Cumlick ~C(Condition, Treatment(reference='Control'))*Trial", data, groups=data['Animal'])
    result = model.fit()
    return result.summary()
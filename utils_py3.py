#### OVERALL summary of code:
'''
This code defines various functions for loading, processing, and analyzing data. Here's a summary of what each function does:

load_all_animals_vars: Loads a variable from a specific session for all animals into a dictionary.
load_variable: Loads a variable from a file.
load_cellreg_dict: Loads cell registration data from a JSON file.
read_behavior: Reads behavior data from a file. ********NOTE: there is a version for Linux/Unix, and one for Windows ********
parse_behavior: Parses behavior data based on a specific text pattern.
filter_cycle: Filters data based on a cycle.
compute_all_dffs: Computes dF/F for all cycles.
compute_mean_traces: Computes mean traces.
compute_baseline: Computes baseline.
compute_auc_tone: Computes area under the curve (AUC) during the tone period.
compute_auc_period: Computes AUC during a specific period.
compute_auc_pretone: Computes AUC before the tone period.
compute_auc_posttone: Computes AUC after the tone period.
extract_single_cycle: Extracts data for a single cycle.
parse_cells_from_filename: Parses cells from a filename.
compute_significance: Computes significance.
load_data_from_mat: Loads data from a .mat file.
calculate_standard_scores: Calculates standard scores.
load_and_split_data: Loads and splits data into training and testing sets.
cross_val_accuracy: Computes cross-validation accuracy.
compute_empirical_covariance: Computes empirical covariance.
save_variable: Saves a variable to a file.
compute_adjusted_pvalues: Computes adjusted p-values.
main: Main function to load, preprocess, and analyze data.
The main function loads data from .mat files, splits it into training and testing sets, trains a Support Vector Classifier (SVC) model, and prints the training, test, and cross-validation accuracies.

'''

#!/usr/bin/env python
# coding: utf-8

import re
import os
import pickle
import xml.etree.ElementTree as ET
import json

import numpy as np
import matplotlib.pyplot as pl
from scipy import stats as sstats
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.covariance import EmpiricalCovariance
from sklearn.svm import SVC
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import cosine_similarity

def load_all_animals_vars(varname, animal_list, session, notebook='preprocessing', func=None):
    '''
    A convenience function to load a given variable from a specific session for all animals into a dictionary.
    Example:
        >>> import utils as ut
        >>> vars_dict = load_all_animals_vars('events', ['calvin', 'darla', 'danny'], 'sepodor_pre', 'preprocessing')

    The func argument can be used to apply a lambda function to the loaded variables.
    It's useful for example for extracting number of cells from the traces data.
    Example:
        >>> import utils as ut
        >>> vars_dict = ut.load_all_animals_vars('traces', ['calvin', 'darla', 'danny'], 'sepodor_pre', 'preprocessing',
        ... func=lambda x: x.shape[1])
    '''
    if func == None:
        func = lambda x: x
    all_of_them = {}
    for ani in animal_list:
        folder = '../%s/%s/notebooks/autorestore/%s/' % (ani, session, notebook)
        try:
            all_of_them.update({ani: func(load_variable(varname, folder))})
        except IOError:
            print('Could not load: %s for %s in %s/%s' % (varname, ani, session, notebook))
    return all_of_them

def load_variable(name, folder='./'):
    with open(os.path.join(folder, name),'rb') as f: # JSB changed this from with open(os.path.join(folder, name),'rb') as f: 5/31/24
        toret = pickle.load(f)
    return toret

def load_cellreg_dict(name, session):
    with open(name, 'r') as f:
        toret = json.load(f)
    return np.r_[toret[session]][np.nonzero(np.prod(np.c_[[v for v in toret.itervalues()]], 0))]-1

def read_behavior(filename, sync_to_begin=False, begin_string='BEGIN'):
    try:
        with open(filename, encoding='latin-1') as behavior_file: #6/24 -added encoding='latin-1' because throwing UTF-8 endcoding error when ran code on ubuntu/Unix, but not windows (python3). Seems like this is due to default coding in windows and unix being different. So, if running in windows, don't technically need to add "encoding='latin-1'"
            behavior = behavior_file.readlines()
    #         behavior = [[int(bb) for bb in b[:-2].split()] for b in behavior if b[0] != '#' and b[0] != '\r']
            behavior = [b.split() for b in behavior if b[0] != "#" and b[0] != "\n"] #I believe if using windows, change /n to /r
            behavior = [[float(b[0])*1.e-3, b[1]] for b in behavior]
    except (IOError):
        print("Wait, I wasn't able to open the behavior.txt file, did you create one?")
        raise
    if sync_to_begin:
        start_2p = parse_behavior(behavior, begin_string)[0]
        behavior = [[float(b[0])-float(start_2p), b[1]] for b in behavior]  
    return behavior

def read_behavior_windows(filename, sync_to_begin=False, begin_string='BEGIN'): #differs from read_behavior fxn above by not specifying 'latin-1' encoding and also changing "\n" to "\r" - those settings are for when python is run in Unix
    try:
        with open(filename) as behavior_file:
            behavior = behavior_file.readlines()
            behavior = [b.split() for b in behavior if b[0] != "#" and b[0] != "\r"]
            behavior = [[float(b[0])*1.e-3, b[1]] for b in behavior]
    except (IOError):
        print("Wait, I wasn't able to open the behavior.txt file, did you create one?")
        raise
    if sync_to_begin:
        start_2p = parse_behavior(behavior, begin_string)[0]
        behavior = [[float(b[0])-float(start_2p), b[1]] for b in behavior]  
    return behavior

def parse_behavior(behavior, text, offset=0):
    p = re.compile(text)
    return np.r_[[b[0]+offset for b in behavior if p.search(b[1])]]

def filter_cycle(time_ax, cycles, cycle):
    return (time_ax>=cycles[cycle][0]) * (time_ax<cycles[cycle][1])

def compute_all_dffs(time_ax, dff, cell, cycles, time_ax_single, cycle_filter=lambda x: True):
    return np.r_[[dff[:, cell][filter_cycle(time_ax, cycles, cycle)][:len(time_ax_single)]
                              for cycle in range(len(cycles)) if cycle_filter(cycle)]]

def compute_mean_traces(time_ax, dff, cycles, time_ax_single, cycle_filter=lambda x: True):
    traces_means = np.zeros((len(time_ax_single), dff.shape[1]))
    traces_std = np.zeros((len(time_ax_single), dff.shape[1]))
    for cell in range(dff.shape[1]):
        all_dffs = compute_all_dffs(time_ax, dff, cell, cycles, time_ax_single, cycle_filter)
        traces_means[:, cell] = all_dffs.mean(0)
        traces_std[:, cell] = np.std(all_dffs, 0)
    return traces_means, traces_std

def compute_baseline(time_ax, dff, base_start, base_stop):
    base_bool = (time_ax>=base_start) * (time_ax<base_stop)
    return dff[base_bool].mean(0)

def compute_auc_tone(time_ax, dff, cycles, time_ax_single, auc_baseline,
    tone_start=0, tone_duration=30):
    traces_means, traces_std = compute_mean_traces(time_ax, dff, cycles, time_ax_single)
    fps = 1./np.diff(time_ax)[0]
    fc = (time_ax_single>=tone_start) * (time_ax_single<tone_duration)    
    return np.sum((traces_means[fc]-auc_baseline), 0)/(4*fps)

def compute_auc_period(time_ax, dff, cycles, cycle, time_ax_single, cell, cycle_start=-10, start=0, end=30):
    t, dff_single = extract_single_cycle(time_ax, dff, cycles, cycle, cell, cycle_start=cycle_start)
    fps = 1./np.diff(time_ax)[0]
    fc = (time_ax_single>=start) * (time_ax_single<end)
    return np.sum(np.r_[dff_single][np.where(fc)])/fps

def compute_auc_pretone(time_ax, dff, cycles, time_ax_single, auc_baseline,
    tone_start=0, tone_duration=30, pretone_duration=30):
    traces_means, traces_std = compute_mean_traces(time_ax, dff, cycles, time_ax_single)
    fps = 1./np.diff(time_ax)[0]
    fc = (time_ax_single>=(tone_start-pretone_duration)) * (time_ax_single<tone_start)
    return np.sum((traces_means[fc]-auc_baseline), 0)/(4*fps)

def compute_lick_ratios(licks, cycles, cycle_start=-10, cs_start=0, cs_end=4, delay=4,
                        cs_duration=4, zero_value=-1):
    lick_ratios = []
    for s, e in cycles:
        l = licks - s + cycle_start
        licks_during = ((l>cs_start)*(l<cs_end+delay)).sum()
        licks_all = ((l>-cs_duration-delay)*(l<cs_end+delay)).sum()
        lick_ratios.append(1.*licks_during/licks_all if licks_all>0 else zero_value)
    return np.r_[lick_ratios]

def compute_licks_during(licks, cycles, start=-10, end=20):
    licks_during = []
    for s, e in cycles:
        l = licks - s
        licks_during.append(((l>start)*(l<end)).sum())
    return np.r_[licks_during]

def compute_licks_rate_during(licks, cycles, start=-10, end=20):
    licks_during = []
    for s, e in cycles:
        l = licks - s
        licks_during.append(((l>start)*(l<end)).sum()/(end-start+1))
    return np.r_[licks_during]

def compute_lick_rate(licks, time_ax):
    lick_rate = np.zeros_like(time_ax)
    for l in licks:
        lick_rate[np.argmin(abs(l-time_ax))] += 1
    return lick_rate

def extract_single_cycle(time_ax, dff, cycles, cycle, cell,
                         cycle_start=-10):
    fc = filter_cycle(time_ax, cycles, cycle)
    t0 = time_ax[fc][0]
    return time_ax[fc] - t0 + cycle_start, dff[:, cell][fc]


def extract_single_cycle_signal(time_ax, signal, cycles, cycle, cycle_start=-10):
    fc = filter_cycle(time_ax, cycles, cycle)
    t0 = time_ax[fc][0]
    return time_ax[fc] - t0 + cycle_start, signal[fc]

def extract_single_cycle_time_ax(time_ax, cycles, cycle_duration=4, cycle_start=-10):
    # single_cycle_time_bins = filter_cycle(time_ax, cycles, cycle).sum() 
    min_len = np.inf
    truncated = 0
    ls = []
    for i, c in enumerate(cycles):
        time_ax_single = time_ax[filter_cycle(time_ax, cycles, i)]-time_ax[filter_cycle(time_ax, cycles, i)][0]
        time_ax_single = time_ax_single[time_ax_single < cycle_duration] + cycle_start
        ls.append(len(time_ax_single))
    if len(np.unique(ls))>1:
        print("Warning: I found cycles with different time bin lengths. Plus-minus one frame is generally ok.")
        print(ls)
    else:
        print("length of cycles",ls)
    return time_ax_single[:np.min(ls)]

def eliminate_cycles(cycles, which_ones):
    return np.r_[[c for i, c in cycles if i not in which_ones]]

def get_cycles_durations(cycles, time_ax):
    return zip(np.diff(cycles, 1).flatten(),
               [filter_cycle(time_ax, cycles, i).sum() for i in range(len(cycles))])

def get_available_strings(behavior):
    return np.unique([b[1] for b in behavior]).tolist()

def read_time_ax_xml(xmlfile, sync_to_begin=False):
    # grab time axis from the xml file

    print("I infer the time axis from:\n", xmlfile)
    tree = ET.parse(xmlfile)
    root = tree.getroot()
   
    time_ax = np.r_[[child.attrib['absoluteTime']
                  for child in root.iter('Frame')]].astype(float)

    if sync_to_begin:
        time_ax -= time_ax[0]

    return time_ax

def search_cycle(behavior, cycles, event):
    event_times = parse_behavior(behavior, event)
    return [any(map(lambda t: (t>=s) and (t<e), event_times)) for s, e in zip(cycles[:, 0], cycles[:, 1])]
        
def search_events(cycles, cycle, event_times):
    cycle_start, cycle_end = cycles[cycle]
    return np.r_[(event_times>=cycle_start) * (event_times<cycle_end)]

def compute_mean_level(time_ax, dff, cycles, cycle, cell, start, end, cycle_start=-10):
    t, tr = extract_single_cycle(time_ax, dff, cycles, cycle, cell, cycle_start=-10)
    return tr[(t>=start) * (t<end)].mean()

def zscore_traces(dff):
    return StandardScaler().fit_transform(dff)

def resample_signal(original_signal, original_time_ax, new_time_ax):
    return np.interp(new_time_ax, original_signal, original_time_ax)

def event_detection_cnmfe_denoised(C):
    return np.r_[[np.clip(np.r_[0, np.diff(C[:, cell])], 0, np.inf) for cell in range(C.shape[1])]].T

def events_to_array(events, time_ax):
    array = np.zeros_like(time_ax)
    for e in events:
        array[np.argmin(abs(time_ax-e))] += 1
    return array

def convert_p_in_stars(values, significances=[0.05, 0.01, 0.001]):
    return np.r_[["***" if r<significances[2] else
                  "**" if r<significances[1] else
                  "*" if r<significances[0] else
                  None
                 for r in values]]

def combine_cycles(time_ax, dff, cycles, cell, lim_len=-1):
    max_cycles = len(cycles)
    return np.r_[[dff[:, cell-1][filter_cycle(time_ax, cycles, cycle)][:lim_len]
                  for cycle in range(max_cycles)]]

def time_to_first_event_in_cycle(cycles, behavior, event="REWARD", which_cycles=None):
    """
    Use which_cycles (a boolean list) to apply the function only to some cycles,
    for example:
        rewards = np.r_[ut.parse_behavior(behavior, 'REWARD')]
        is_rewarded = [any(map(lambda r: (r<e)*(r>=s), rewards))
                       for s, e in cycles]
        time_to_first_event_in_cycle(cycles, "REWARD", is_rewarded)
    """
    if which_cycles == None:
        which_cycles = np.r_[[True] * len(cycles)]
    event_times = np.r_[parse_behavior(behavior, event)]
#     first_event_times = np.r_[[event_times[np.where((event_times-s)>0)[0][0]]-s + CYCLE_START
#                                for s, e in cycles[np.where(which_cycles)]]]
    first_event_times = []
    for s, e in cycles[np.where(which_cycles)]:
        try:
            w = np.where((event_times-s)>0)[0][0]
            w_time = event_times[w]-s + CYCLE_START
            if w_time < CYCLE_DURATION:
                first_event_times.append(event_times[w]-s + CYCLE_START)
            else:
                first_event_times.append(-1)
        except:
            first_event_times.append(-1)
    return np.r_[first_event_times]

def shift_cycles(cycles, shifts):
    return cycles[shifts > 0] - shifts[shifts > 0][:, None]


def combine_cells(patterns_list, labels_list, train_size=0.8, patterns_per_label=100):
    all_data = [train_test_split(p, l, train_size=train_size, stratify=l)
                for p, l in zip(patterns_list, labels_list)]

    dp_training = []
    dl_training = []
    dp_test = []
    dl_test = []
    for label in np.unique(labels_list[0]):
        for i in range(patterns_per_label):
            choices = [np.random.choice(np.where(l==label)[0]) for p, P, l, L in all_data]
            dp_training.append(np.concatenate([p[c] for (p, P, l, L), c in zip(all_data, choices)]))
            dl_training.append(label)
        for i in range(patterns_per_label):
            choices = [np.random.choice(np.where(L==label)[0]) for p, P, l, L in all_data]
            dp_test.append(np.concatenate([P[c] for (p, P, l, L), c in zip(all_data, choices)]))
            dl_test.append(label)

    dp_training = np.r_[dp_training]
    dl_training = np.r_[dl_training]
    dp_test = np.r_[dp_test]
    dl_test = np.r_[dl_test]
    
    return dp_training, dp_test, dl_training, dl_test

def load_spatial_footprints(coor_file, cnn_file=None, key='Sources2D'):
    mean_image = np.loadtxt(cnn_file) if cnn_file is not None else None
    contours = loadmat(coor_file)[key][:, 0]
    return mean_image, contours

def load_spatial_footprints_A(A_file, shape=(512, 512)):
    return np.loadtxt(A_file).T.reshape([-1, shape[0], shape[1]])

def combine_patterns(patterns, labels, n_patterns=100, classes=[0, 1], labels_mask=None):
    """
    For all cells in each individ animal, fxn randomly pulls a trial (within specified trial type). Will do this 'n_patterns' times, such that each pattern produces a new combination of trials included across animals. Thus, you will lose info that may be embedded in different time epochs of the overall session (ie, beginning, middle, end of session).
    """
    labels_comb = np.r_[list(classes)*n_patterns]
    patterns_comb_train = []
    for i in range(n_patterns):
        for odor in classes:
            p = np.concatenate([patterns[ani][np.random.choice([w for w in np.where(labels[ani]==odor)[0]])]
                                for ani in patterns.keys()])
            patterns_comb_train.append(p) #p will be length n cells across all animals (pseudopopulation size)
    return np.r_[patterns_comb_train], labels_comb

def compute_selectivity(time_ax, activity, cycles, timeframe, baseline_timeframe, stat_func=None, **stat_func_args):
    """
    timeframe and baseline_timeframe can be (START, STOP) times with respect to cycle start
    or a list of (START, STOP) times for each cycle.
    """
        
    labels_time_ax = np.zeros_like(time_ax)
    
    if timeframe == None:
        for s, e in cycles:
            labels_time_ax[(time_ax>=s) * (time_ax<e)] = 1
        
    elif len(timeframe) == 2:   
        for i, (s, e) in enumerate(cycles):
            labels_time_ax[(time_ax>=(s+timeframe[0])) *
                           (time_ax<(s+timeframe[1]))] = 1
    elif len(timeframe) == len(cycles):
        for i, ((s, e), t) in enumerate(zip(cycles, timeframe)):
            labels_time_ax[(time_ax>=(s+t[0])) *
                           (time_ax<(s+t[1]))] = 1
    else:
        raise Exception('Length of timeframe == %d and len of cycles == %d. Should be the same instead'%
                        (len(timeframe), len(cycles)))

    if len(baseline_timeframe) == 2:   
        for i, (s, e) in enumerate(cycles):
            labels_time_ax[(time_ax>=(s+baseline_timeframe[0])) *
                           (time_ax<(s+baseline_timeframe[1]))] = -1
    elif len(baseline_timeframe) == len(cycles):
        for i, ((s, e), t) in enumerate(zip(cycles, baseline_timeframe)):
            labels_time_ax[(time_ax>=(s+t[0])) *
                           (time_ax<(s+t[1]))] = -1
    else:
        raise Exception('Length of timeframe == %d and len of cycles == %d. Should be the same instead'%
                        (len(baseline_timeframe), len(cycles)))
        
    if stat_func == None:
        stat_func = lambda x, y: sstats.mannwhitneyu(x, y, alternative='two-sided')

    selectivity = []
    for cell in range(activity.shape[1]):
        act_cs = activity[:, cell][labels_time_ax==1]
        act_base = activity[:, cell][labels_time_ax==-1]
        try:
            selectivity.append([stat_func(act_cs, act_base, **stat_func_args),
                                np.sign(np.mean(act_cs)-np.mean(act_base))])
        except ValueError:
            # if activity == 0 in both conditions
            selectivity.append([np.r_[0, 1], 0])

    return np.r_[selectivity]

def adjust_pvalues(pvalues, method='fdr_bh', **method_args):
    return multipletests(pvalues, method=method, **method_args)[1]

def extract_patterns(time_ax, activity, cycles, CYCLE_START, STIM_START, STIM_END, mode='average'):
    n_cells = activity.shape[1]
    if mode == 'average':
        patterns = np.zeros((len(cycles), n_cells))
    elif mode == 'corr':
        cov_model = EmpiricalCovariance()
        patterns = np.zeros((len(cycles), n_cells, n_cells))
    # tas == a replacement for time_ax_single
    for i, (s, e) in enumerate(cycles):
        time_filter = ((time_ax>=(s-CYCLE_START+STIM_START)) * (time_ax<(s-CYCLE_START+STIM_END)))
        if mode == 'average':
            patterns[i] = activity[time_filter].mean(0)
        elif mode == 'corr':
            patterns[i] = cov_model.fit(activity[time_filter]).covariance_
    patterns = patterns.reshape(len(cycles), -1)
    # patterns = patterns[:, which_cells]
    return patterns

def decode(patterns, labels, decoder=None, which_cells=None, n_loops=10, n_jobs=1, cv=10):
    if which_cells == None:
        which_cells = [True] * patterns.shape[1]
    
    if decoder == None:
        decoder = SVC(kernel='linear')

    scores = []
    scores_chance = []

    ps = patterns[:, which_cells]
    ls = labels
    scores = cross_val_score(decoder, ps, ls, cv=cv, n_jobs=n_jobs)
    scores_chance = []
    for i in range(n_loops):
        scores_chance.append(cross_val_score(decoder, ps, np.random.permutation(ls), cv=cv, n_jobs=n_jobs))
    scores_chance = np.r_[scores_chance].flatten()
    return scores, scores_chance

def compute_similarity_matrix(pattern_ids, all_patterns, similarity_func=None):
    if similarity_func == None:
        similarity_func = lambda x, y: sstats.pearsonr(x, y)[0]
    corrmat_distr = {}
    for i, (l, a) in enumerate(zip(pattern_ids, all_patterns)):
        for j, (m, b) in enumerate(zip(pattern_ids, all_patterns)):
            temp = []
            try:
                for ii, aa in enumerate(a):
                    for jj, bb in enumerate(b):
                        if ii==jj or aa.sum()==0 or bb.sum()==0: continue
                        temp.append(similarity_func(aa, bb))
            except ValueError:
                print("Cannot compute similarity between %s and %s." % (l, m))
            corrmat_distr[(l, m)] = temp
    corrmat = np.zeros((len(pattern_ids), len(pattern_ids)))
    for i, p in enumerate(pattern_ids):
        for j, q in enumerate(pattern_ids):
            corrmat[i][j] = np.mean(corrmat_distr[(p, q)])
    return corrmat_distr, corrmat            

def compute_mean_activity_patterns(time_ax, activity, cycles, timeframe):
    start, stop = timeframe
    return np.r_[[np.mean(activity[(time_ax >= (s+start)) * (time_ax < (s+stop))], 0)
                 for s, e in cycles]]

def compute_similarity_matrix_woods(pattern_ids, all_patterns):   
    corrmat_distr = {}
    for i, (l, a) in enumerate(zip(pattern_ids, all_patterns)):
        for j, (m, b) in enumerate(zip(pattern_ids, all_patterns)):
            temp = []
            for ii, aa in enumerate(a):
                for jj, bb in enumerate(b):
                    # skip if same vector or any of the 2 == zero
                    if ii==jj or np.sum(aa)==0 or np.sum(bb)==0: continue
                    # count once if using same data
                    if l==m and jj<ii: continue
                    temp.append(sstats.pearsonr(aa, bb)[0])
            corrmat_distr[(l, m)] = temp
    corrmat = np.zeros((len(pattern_ids), len(pattern_ids)))
    for i, p in enumerate(pattern_ids):
        for j, q in enumerate(pattern_ids):
            corrmat[i][j] = np.mean(corrmat_distr[(p, q)])
    return corrmat_distr, corrmat
           
def extract_traces_around_event(time_ax, traces, evs, tpre, tpost):
    extracted = [traces[(time_ax>=(e-tpre))*(time_ax<(e+tpost))]
                        for e in evs]
    min_len = np.min([len(t) for t in extracted])
    return np.r_[[t[:min_len] for t in extracted]]

def extract_activity(time_ax, activity, cycles, CYCLE_START, STIM_START, STIM_END,
                     offset=0, which=None):
    if which == None:
        which = [True] * len(cycles)
    return np.r_[[activity[(time_ax>=(start-CYCLE_START+STIM_START+offset))
                           *(time_ax<(start-CYCLE_START+STIM_END+offset))].mean(0)
                  for start, stop in cycles[which]]]

def generate_combined_cells_patterns(patterns_dict, labels_dict, n_patterns=30, labels=None, animals=None):
    if labels == None:
        labels = np.unique(np.concatenate(labels_dict.values()))
    if animals == None:
        animals = patterns_dict.keys()
    patterns_combined = []
    for o in labels:
        patterns_o = []
        for m, v in patterns_dict.iteritems():
            patterns_o.append(np.c_[[np.random.choice(v[labels_dict[m]==o][:, cell], size=n_patterns)
                                     for cell in range(v.shape[1])]].T)
        patterns_combined.append(np.column_stack(patterns_o))
    patterns_combined = np.row_stack(patterns_combined)

    labels_combined = np.r_[[[o]*n_patterns for o in labels]].flatten()
    return patterns_combined, labels_combined

def sig_95(vals):
    return [0, 0 if ((np.mean(vals)-50)/(sstats.sem(vals)*2))>=1 else 1]

def sync_behavior_to_xml(time_ax, behavior, delta_t_min=0.4, piezo=False, code_begin='BEGIN', code_end='END'):
    """
    Changes the behavior times to match the xml times based on when the timings of the recordings
    and the arduino begin and end codes. The timings are extimated from xml by looking at the distances
    between consecutive frames. If the distance is larger than `delta_t_min`, a recording cycle is
    inferred at that time and a code in arduino is expected at the same time (`cycle_code`).
    Since an equal number of cycles and arduino codes are expected, the time differences between
    corresponding times are used to rescale the arduino cycles and all the events therein.
    Events recorded in between cycles (e.g., licks) are rescaled using the previous trial.
    Events recorded before the beginning of the first cycle are not rescaled.
    
    ***JSB edit 4/2/20, added special case for continuous imaging (if(len)cycles_xml ==1). Also, it seems like the 
    end time for the last cycle in cycle imaging scenario is not adjusted***
    ***JSB edit 5/10/23 added piezo variable, which if 'True' will eliminate elongated scaling of arduino (which is almost certainly 
    erroneously applied due to imaging shut off being delayed w.r.t. arduino off trigger when using piezo)
    
    Arguments
    =========
    
        time_ax : 1-d array
            The xml-inferred time axis to which behavior will be aligned to
            
        behavior : 2-d array of shape (n_events, 2)
            Array of time stamps and corresponding behavior codes
        
        delta_t_min : float
            Minimum distance in seconds between consecutive frames to be considered a cycle
            
        code_begin : string
            The code in the behavior array that identifies a the start of a cycle
        code_end : string
            The code in the behavior array that identifies a the end of a cycle
    """
#     delta_t_min = 0.4
    delta_ts = np.diff(time_ax)
    try:
        cycle_starts_xml = time_ax[np.r_[0, np.where(delta_ts>delta_t_min)[0]+1]]
        cycle_ends_xml = time_ax[np.r_[np.where(delta_ts>delta_t_min)[0], len(time_ax)-1]]    
    except IndexError:
        cycle_starts_xml = time_ax[0]
        cycle_ends_xml = time_ax[-1]
    cycles_xml = np.c_[cycle_starts_xml, cycle_ends_xml]
    cycle_starts_ard = np.sort(parse_behavior(behavior, code_begin))
    cycle_ends_ard = np.sort(parse_behavior(behavior, code_end))
    try:
        cycles_ard = np.c_[cycle_starts_ard, cycle_ends_ard]
    except ValueError:
        raise Exception("The number of begin and end events in arduino don't match.")
    # pl.vlines(cycle_starts_xml, 0, 1)
    # pl.vlines(cycles_starts_ard, 1, 2, color='r')
    # pl.xlim(-10, 200)
    # pl.title('Cycle Starts')
    # pl.legend(['XML', 'ARDUINO'])
    # pl.xlabel('Time (s)')
    # pl.yticks(())
    
    if len(cycles_xml) != len(cycles_ard):
        raise Exception("Number of cycles in XML and Arduino file don't match, try changing delta_t_min.")
    diffs = np.diff(cycles_xml, 1).flatten()-np.diff(cycles_ard, 1).flatten()
    factors = 1+diffs/np.diff(cycles_ard, 1).flatten()
    print(factors)
    #adding this for piezo imaging because noticed that imaging shutoff often lagged arduino trigger, thus making it seem like arduino clock == faster than xml clock, when in reality it was not. so if this code wants to elongate arduino time, manually cancel that
    if piezo==True:
        for i,x in zip(factors,range(len(factors))):
            if i > 1:
                factors[x]=1
        print('manually adjusted factors due to lag in imaging shut off when using piezo',factors)
    
    new_times = np.r_[[b[0] for b in behavior]]
    newer_times=np.ones(len(behavior))*-1   # Added this 10/23/20 to fix issue with some rare events being pushed into the subsequent cycle
    
    if len(cycles_xml)>1:
        for f, s1, s2, sxml in zip(factors, cycles_ard[:, 0], cycles_ard[:, 0][1:], cycles_xml[:, 0]):
            filt = (new_times>=s1) * (new_times<s2)
            newer_times[filt] = (new_times[filt]-new_times[filt][0])*f + sxml
        # last one
        filt = new_times>=cycles_ard[-1][0]
        newer_times[filt] = (new_times[filt]-new_times[filt][0])*f+cycles_xml[-1][0]
        return [[nt, b[1]] for nt, b in zip(newer_times, behavior) if nt >= 0]
    elif len(cycles_xml) ==1:
        for s1, s2, sxml in zip(cycles_ard[:, 0], cycles_ard[:, 0][1:], cycles_xml[:, 0]):
            filt = (new_times>=s1) * (new_times<s2)
            new_times[filt] = (new_times[filt]-new_times[filt][0])*factors + sxml
        # last one
        filt = new_times>=cycles_ard[-1][0]
        new_times[filt] = (new_times[filt]-new_times[filt][0])*factors+cycles_xml[-1][0]
        return [[nt, b[1]] for nt, b in zip(new_times, behavior)]
    
    
#####################################################################
###################### JSB ADDITIONS 4/2024+ ########################
#####################################################################

import matplotlib
from matplotlib.lines import Line2D

def add_significance(ax, array1, array2, x1, x2, y, ticksize=0.02, sig_func=None, thresholds=(0.05, 0.01, 0.001),mcc=1,suppress_ns=True):
    #mmc = multiple-comparisons correction (using bonferroni), where mcc = number of comparisons))
    """
    fxn tests whether two data sets are significantly different and plot a line with p-value
    """
    if sig_func is None:
        sig_func = lambda x, y: sstats.mannwhitneyu(x, y, alternative='two-sided')
    elif sig_func == 'ttest':
        sig_func = lambda x, y: sstats.ttest_ind(x,y)
    p = sig_func(array1, array2)[-1]
    sig_value = sig_func(array1, array2)[0]
    deltay = (np.diff(ax.axis()[-2:])*ticksize)[0]#had to add '[0]' at end so that delta y was a value, not a list
    if suppress_ns==True:
        if p <= thresholds[0]/mcc:
            if x1!=x2:
                line = Line2D([x1, x1, x2, x2], [y-deltay, y, y, y-deltay], lw=.5, color='k', clip_on=False)
                ax.add_line(line)
            ax.text(np.mean([x1, x2]), y,
                    '*' if p > thresholds[1]/mcc else
                    '**' if p > thresholds[2]/mcc else
                    '***',
                    ha='center', fontsize=5)
    else:
        if x1!=x2:
            line = Line2D([x1, x1, x2, x2], [y-deltay, y, y, y-deltay], lw=.5, color='k', clip_on=False)
            ax.add_line(line)
        ax.text(np.mean([x1, x2]), y+1,
            'n.s.' if p > thresholds[0]/mcc else
            '*' if p > thresholds[1]/mcc else
            '**' if p > thresholds[2]/mcc else
            '***',
            ha='center', fontsize=5)
    return sig_value, p


def add_significance_point(ax, array1, array2, x1, y, ticksize=0.02, sig_func=None, thresholds=(0.05, 0.01, 0.001),
                                   mcc=1,color='k',symbol='*'):
    """
    fxn tests whether two data sets are significantly different and plots symbol if significant
    """
    if sig_func == None:
        sig_func = lambda x, y: sstats.mannwhitneyu(x, y, alternative='two-sided')
    elif sig_func == 'ttest':
        sig_func = lambda x, y: sstats.ttest_1samp(x,y)
    p = sig_func(array1, array2)[-1]
    sig_value = sig_func(array1, array2)[0]
    deltay = np.diff(ax.axis()[-2:])*ticksize
    ax.text(x1, y,
            "" if p > thresholds[0]/mcc else
            symbol if p > thresholds[1]/mcc else
            symbol+symbol if p > thresholds[2]/mcc else
            symbol+symbol+symbol,
            #symbol if p < thresholds[1]/mcc else
            #"",
            ha='center', fontsize=5,color=color)
    return sig_value, p


def report_stats (array1, array2,loops,sig_func=None,mcc=1):
    if sig_func == None:
        sig_func = lambda x, y: sstats.mannwhitneyu(x, y, alternative='two-sided')
        U = sig_func(array1, array2)[0]
        z = ((U-((loops*loops)/2))/(np.sqrt(((loops*loops)*(loops+loops+1))/12)))
        r = z/np.sqrt(loops*2)
        p = sig_func(array1, array2)[-1]
        print("U =",U,"p =",p*mcc,"r =",r)
    if sig_func == 'ttest':
        sig_func = lambda x, y: sstats.ttest_ind(x, y)
        cohens_d = (np.mean(array1)-np.mean(array2))/np.sqrt((np.std(array1)**2+np.std(array2)**2)/2)
        p = sig_func(array1, array2)[-1]
        print(sig_func(array1, array2),"d =", cohens_d, 'corrected p =',p*mcc)
    if sig_func == 'fishers':
        table = array1,array2
        sig_func = lambda x: sstats.fisher_exact(x, alternative='two-sided')
        p = sig_func(table)[-1]
        odds_ratio = sig_func(table)[0]
        print(sig_func(table),"odd's ratio =", odds_ratio, 'corrected p =',p*mcc)
        
        
def combine_train_test_patterns(patterns, labels, train_test_split, classes=None, n_cells=None,
                                relabel=None, relabel_test=None, which_trials=None):
    """
patterns is your imaging data broken into trials (an array of shape: trials, cells); labels the label/class designation for each trial     (so decoder knows which imaging data is associated with witch trial type); classes is which trial type labels you want to use for decoding/
comparing; n_cells is how many cells you want to use for decoding; train_test_split is the ratio of trials for each label you want to
use for traning or testing; relabel/relabel_test is used if you want to combine distinct labels under a similar label (used
primarily for CCGP); which_trials I use if I want to restrict decoding to a subset of trials of each trail type.
    """
    if classes == None:
        classes = [0, 1] #default

    which_train = {}
    which_test = {}
    for ani in patterns.keys():
        which_trains = []
        which_tests = []
        
        if which_trials == 'first_10':
            #print('using first 10 trials',)
            for l in classes:
                if l>=0: #labels with value less than zero will be ignored
                    #grab randomized order of 1st 10 trials of this trial type. split these 10 trials for train and test
                    wt = np.random.permutation(np.where(labels[ani]==l)[0][:10])
                    if train_test_split<1:
                        wT = wt[:5]
                        which_trains.append(wt[5:10])
                        which_tests.append(wT)
                    else:
                        which_trains.append(wt)
        elif which_trials == 'last_10':
            #print('using last 10 trials',)
            for l in classes:
                if l>=0:
                    wt = np.random.permutation(np.where(labels[ani]==l)[0][-10:])
                    if train_test_split<1:
                        wT = wt[:5]
                        which_trains.append(wt[5:10])
                        which_tests.append(wT)
                    else:
                        which_trains.append(wt)
        elif which_trials == 'any_10':
            #print('using 10 random trials',)
            for l in classes:
                if l>=0:
                    wt = np.random.permutation(np.where(labels[ani]==l)[0])
                    if train_test_split<1:
                        wT = wt[:5]
                        which_trains.append(wt[5:10])
                        which_tests.append(wT)
                    else:
                        which_trains.append(wt)
        elif which_trials == 'min':
            #if your total number of trials differs across trial type, and you want to downsample to the lowest
            min_trials = np.min([len(np.where(labels[ani]==l)[0]) for l in classes])
            for l in classes:
                if l>=0:
                    wt = np.random.permutation(np.where(labels[ani]==l)[0][:min_trials])
                    if train_test_split<1:
                        which_trains.append(wt[int(len(wt)*train_test_split):])
                        which_tests.append(wt[:int(len(wt)*train_test_split)])
                    else:
                        which_trains.append(wt)
        else:
            #print('using all trials',)
            for l in classes:
                if l>=0:
                    # extract ALL trials of a trial type, in random order, then split them into train and test (with ratio determied by train_test_split value)
                    wt = np.random.permutation(np.where(labels[ani]==l)[0])
                    if train_test_split<1:
                        wT = wt[:int(len(wt)*train_test_split)]
                        which_trains.append(wt[int(len(wt)*train_test_split):])
                        which_tests.append(wT)
                    else:
                        which_trains.append(wt)
        which_train[ani] = np.concatenate(which_trains)
        if train_test_split<1:
            which_test[ani] = np.concatenate(which_tests)
    
    #make copy of full patterns and labels data, then slice them to keep only the trials designated above as training data
    patterns_t = patterns.copy()
    labels_t = labels.copy()
    for ani in patterns.keys():
        patterns_t[ani] = patterns[ani][which_train[ani]]
        labels_t[ani] = labels[ani][which_train[ani]]
    #the combine_patterns fxn will, for each trial type (label), independently select a random trial from each animal until the array is
    # populated with 100 trials for each animal (note: it will extract 100 "trials", even if > 100 trials total for a trial type)
    # Having 100 trials allows for a large variety of different trial combinations across animals.
    # the output is a repeating list of labels (or classes) (length = 100 * the no.of labels), and a randomized list of patterns
    # that corresponds to the order of labels in repeating list.
    patterns_comb_train, labels_comb_ = combine_patterns(patterns_t, labels_t, classes=classes)
        
    #do the same for the designated test data
    if train_test_split<1:
        patterns_T = patterns.copy()
        labels_T = labels.copy()
        for ani in patterns.keys():
            patterns_T[ani] = patterns[ani][which_test[ani]]
            labels_T[ani] = labels[ani][which_test[ani]]
        patterns_comb_test, labels_comb_test_ = combine_patterns(patterns_T, labels_T, classes=classes)
    else:
        patterns_comb_test = None
    
    #If you are relabeling the data, that will be done here. Note that if using 'relabel', but not 'relabel_test', both the 
    # training AND testing labels will be relabeled in accordance with 'relabel'
    if relabel is not None:
        labels_comb = np.r_[[relabel[l] for l in labels_comb_]]
    else:
        labels_comb = labels_comb_
    
    if train_test_split<1:
        if relabel_test is not None:
            labels_comb_test = np.r_[[relabel_test[l] for l in labels_comb_test_]]
        else:
            if relabel is not None:
                labels_comb_test = np.r_[[relabel[l] for l in labels_comb_test_]]
            else:
                labels_comb_test = labels_comb_test_
    
# SO I HAD TROUBLE WITH THIS WHEN USING CELLS REGISTERED ACROSS SESSIONS. YOU CAN NOT SEPARATELY SELECT A RANDOM PERMUTATION 
# OF CELLS ACROSS SESSIONS, OR ELSE YOUR DECODER WILL BE USING DIFFERENT IDENTITY CELLS WHEN TRAINING AND TESTING!! USE DIFF FXN IN THAT CASE
    # how many cells to use? If less than all of them, select a random subset of cells
    if n_cells is None:
        which_cells = [True]*patterns_comb_train.shape[1]
    else:
        which_cells = np.random.permutation(range(patterns_comb_train.shape[1]))[:n_cells]
     
    patterns_comb_train = patterns_comb_train[:, which_cells]
    patterns_comb_test = patterns_comb_test[:, which_cells] if train_test_split<1 else patterns_comb_test

    if train_test_split<1:
        return (patterns_comb_train[labels_comb>=0], labels_comb[labels_comb>=0],
            patterns_comb_test[labels_comb_test>=0], labels_comb_test[labels_comb_test>=0])
    else:
        return (patterns_comb_train[labels_comb>=0], labels_comb[labels_comb>=0],_,_)


def decode_within(patterns,labels,decoder=SVC(kernel='linear',decision_function_shape='ovo'),train_test_split=0.5,n_loops=10,m_loops=3, chance_loops=3, **args):
    # This will return 2 lists: scores, and scores based on chance (random permutation of labels). size of each = n_loops
    tot_scores_ = [] #temp variable that contains the results of each individual run (n = m_loops)
    tot_scores_chance_ = []
    tot_scores = [] #output that contains the average of each n_loop (n = n_loops)
    tot_scores_chance = []
    for n in range(n_loops): #how many times to run the loop below? final output returns 1 value for each n_loop
        for m in range(m_loops): #run this loop and return the average of all its loops
            #define your training and testing datasets (x, y, xT, yT)
            x, y, xT, yT = combine_train_test_patterns(patterns, labels, train_test_split, **args)
            decoder.fit(x, y) #train your decoder using the training data you specified for the diff trial types (x and y)
            scores_for = decoder.score(xT,yT) #test classification accuracy using the held-out data you specified
            decoder.fit(xT,yT) #now, reverse the train-test datasets
            scores_rev = decoder.score(x,y)
            tot_scores_.append(np.mean((scores_for,scores_rev))) #get the mean of forward and reverse directions
            #shuffle labels to get chance scores
            temp = []
            temp_rev = []
            for i in range(chance_loops): #because the permuation is pretty variable, run chance decoding w/ more iterations
                decoder.fit(x,np.random.permutation(y))
                temp.append(decoder.score(xT,yT))
                decoder.fit(xT,np.random.permutation(yT))
                temp_rev.append(decoder.score(x,y))
            scores_chance_for = np.mean(temp)
            scores_chance_rev = np.mean(temp_rev)
            tot_scores_chance_.append(np.mean((scores_chance_for,scores_chance_rev)))
        tot_scores.append(np.mean(tot_scores_))
        tot_scores_chance.append(np.mean(tot_scores_chance_))
    # return mean of forward and reverse decoding, for both real and chance scenarios
    return tot_scores,tot_scores_chance


def decode_across(patterns1, labels1, patterns2, labels2, train_test_split, decoder=SVC(kernel='linear',decision_function_shape='ovo'), n_loops=10, m_loops=5, chance_loops=5, **args):
    #patterns1 and labels1 should be from your first session, 2 from your second session (or 1 and 2 could be from diff time bins across sessions, etc). This finds distinct patterns and labels for each session. TIP: if decoding across diff sessions, can set train_test_split=1 and feed all the trials for training (or testing) for each session (instead of only half, as we usually do when decoding within a session, ie train_test_split=0.5). That said, if want to compare these results to decode within, should keep train_test_split value the same in both. TAKE NOTE: IF USING THIS FXN TO DECODE ACROSS TIME BINS, THE TRIALS INCLUDED IN TB1 AND TB2 WILL SHOW SOME OVERLAP. IF WANT TRIALS IN TB1 AND TB2 TO BE EXCLUSIVE TO EACH, USE THE decode_within across_tbs FUNCTION THAT'S DOWN BELOW.
    tot_scores_ = [] #temp variable that contains the results of each individual run (n = m_loops)
    tot_scores_chance_ = []
    tot_scores = [] #output that contains the average of each n_loop (n = n_loops)
    tot_scores_chance = []
    for n in range(n_loops):
        for m in range(m_loops): #run this loop and return the average of all loops
        #define your training and testing datasets. will pull one dataset from session 1's patterns and labels, and other from session 2's
            x, y, _, _ = combine_train_test_patterns(patterns1, labels1, train_test_split, **args)
            xT, yT, _, _ = combine_train_test_patterns(patterns2, labels2, train_test_split, **args)
            decoder.fit(x, y) #train your decoder using the training data you specified for the diff trial types (x and y)
            scores_for = decoder.score(xT,yT) #test classification accuracy using the held-out data you specified
            decoder.fit(xT,yT) #now, reverse the train-test datasets
            scores_rev = decoder.score(x,y)
            tot_scores_.append(np.mean((scores_for,scores_rev))) #get the mean of forward and reverse directions
            #shuffle labels to get chance scores
            temp = []
            temp_rev = []
            for i in range(chance_loops): #because the permuation is pretty variable, run chance decoding w/ more iterations
                decoder.fit(x,np.random.permutation(y))
                temp.append(decoder.score(xT,yT))
                decoder.fit(xT,np.random.permutation(yT))
                temp_rev.append(decoder.score(x,y))
            scores_chance_for = np.mean(temp)
            scores_chance_rev = np.mean(temp_rev)
            tot_scores_chance_.append(np.mean((scores_chance_for,scores_chance_rev)))
        tot_scores.append(np.mean(tot_scores_))
        tot_scores_chance.append(np.mean(tot_scores_chance_))
    # return mean of forward and reverse decoding, for both real and chance scenarios
    return tot_scores,tot_scores_chance


#for confusion matrix (use predict instead of score)
def predict_within(patterns, labels,train_test_split=0.5, decoder=SVC(kernel='linear',decision_function_shape='ovo'), **args):
    x, y, xT, test_labels = combine_train_test_patterns(patterns, labels, train_test_split, **args)
    decoder.fit(x, y)
    scores = decoder.predict(xT)
    decoder.fit(x,np.random.permutation(y))
    scores_chance = decoder.predict(xT)
    return scores, scores_chance, test_labels #BTW, y and test labels are the same


def predict_across(patterns1, labels1, patterns2, labels2, train_test_split,  decoder=SVC(kernel='linear', decision_function_shape='ovo'), chance_loops=5, **args):
    #Can set train_tet_split here to 1 if want to use all trials within a session
    x, y, _, _ = combine_train_test_patterns(patterns1, labels1, train_test_split, **args)
    xT, yT, _, _ = combine_train_test_patterns(patterns2, labels2, train_test_split, **args)
    decoder.fit(x, y)
    scores = decoder.predict(xT)
    decoder.fit(xT,yT)
    scores_rev = decoder.predict(x)
    
    decoder.fit(x,np.random.permutation(y))
    scores_chance = decoder.predict(xT)
    decoder.fit(xT,np.random.permutation(yT))
    scores_chance_rev = decoder.predict(x)
    return np.append(scores,scores_rev), np.append(scores_chance,scores_chance_rev), np.append(yT,y)


def combine_train_test_patterns_across_tbs(patterns_tb1, patterns_tb2, labels, train_test_split, classes=None, n_cells=None,
                                relabel=None, relabel_test=None, which_trials=None):
    ''' This is explicitly for decoding across time bins within the same session. The only real diff here from combine_train_test_patterns
        is that we use diff patterns for training and testing (corresponding to tb1 and tb2, respectively). We cant use the 
        decode_across function to decode across tbs within a session because there you can (will) get overlapping trials in the 
        training and testing datasets (however, you CAN use that fxn for decoding across tbs AND sessions)'''
    if classes == None:
        classes = [0, 1] #default

    which_train = {}
    which_test = {}
    for ani in patterns_tb1.keys():
        which_trains = []
        which_tests = []
        
        if which_trials == 'first_10':
            #print('using first 10 trials',)
            for l in classes:
                if l>=0: #labels with value less than zero will be ignored
                    #grab randomized order of 1st 10 trials of this trial type. split these 10 trials for train and test
                    wt = np.random.permutation(np.where(labels[ani]==l)[0][:10])
                    if train_test_split<1:
                        wT = wt[:5]
                        which_trains.append(wt[5:10])
                        which_tests.append(wT)
                    else:
                        which_trains.append(wt)
        elif which_trials == 'last_10':
            #print('using last 10 trials',)
            for l in classes:
                if l>=0:
                    wt = np.random.permutation(np.where(labels[ani]==l)[0][-10:])
                    if train_test_split<1:
                        wT = wt[:5]
                        which_trains.append(wt[5:10])
                        which_tests.append(wT)
                    else:
                        which_trains.append(wt)
        elif which_trials == 'min':
            #if your total number of trials differs across trial type, and you want to downsample to the lowest
            min_trials = np.min([len(np.where(labels[ani]==l)[0]) for l in classes])
            for l in classes:
                if l>=0:
                    wt = np.random.permutation(np.where(labels[ani]==l)[0][:min_trials])
                    if train_test_split<1:
                        which_trains.append(wt[int(len(wt)*train_test_split):])
                        which_tests.append(wt[:int(len(wt)*train_test_split)])
                    else:
                        which_trains.append(wt)
        else:
            #print 'using all trials',
            for l in classes:
                if l>=0:
                    # extract all trials of a trial type, in random order, then split them into train and test (with ratio determied by train_test_split value)
                    wt = np.random.permutation(np.where(labels[ani]==l)[0])
                    if train_test_split<1:
                        wT = wt[:int(len(wt)*train_test_split)]
                        which_trains.append(wt[int(len(wt)*train_test_split):])
                        which_tests.append(wT)
                    else:
                        which_trains.append(wt)
        which_train[ani] = np.concatenate(which_trains)
        if train_test_split<1:
            which_test[ani] = np.concatenate(which_tests)
    
    #make copy of full patterns and labels data, then slice them to keep only the trials designated above as training data
    patterns_t = patterns_tb1.copy()
    labels_t = labels.copy()
    for ani in patterns_tb1.keys():
        patterns_t[ani] = patterns_tb1[ani][which_train[ani]]
        labels_t[ani] = labels[ani][which_train[ani]]
    patterns_comb_train, labels_comb_ = combine_patterns(patterns_t, labels_t, classes=classes)
        
    #do the same for the designated test data
    if train_test_split<1:
        patterns_T = patterns_tb2.copy()
        labels_T = labels.copy()
        for ani in patterns_tb2.keys():
            patterns_T[ani] = patterns_tb2[ani][which_test[ani]]
            labels_T[ani] = labels[ani][which_test[ani]]
        patterns_comb_test, labels_comb_test_ = combine_patterns(patterns_T, labels_T, classes=classes)
    else:
        patterns_comb_test = None
    
    #If you are relabeling the data, that will be done here. Note that if using 'relabel', but not 'relabel_test', both the 
    # training AND testing labels will be relabeled in accordance with 'relabel'
    if relabel is not None:
        labels_comb = np.r_[[relabel[l] for l in labels_comb_]]
    else:
        labels_comb = labels_comb_
    
    if relabel_test is not None:
        labels_comb_test = np.r_[[relabel_test[l] for l in labels_comb_test_]]
    else:
        if relabel is not None:
            labels_comb_test = np.r_[[relabel[l] for l in labels_comb_test_]]
        else:
            labels_comb_test = labels_comb_test_
    
    # how many cells to use? If less than all of them, select a random subset of cells
    if n_cells is None:
        which_cells = [True]*patterns_comb_train.shape[1]
    else:
        which_cells = np.random.permutation(range(patterns_comb_train.shape[1]))[:n_cells]
     
    patterns_comb_train = patterns_comb_train[:, which_cells]
    patterns_comb_test = patterns_comb_test[:, which_cells] if train_test_split<1 else patterns_comb_test

    return (patterns_comb_train[labels_comb>=0], labels_comb[labels_comb>=0],
            patterns_comb_test[labels_comb_test>=0], labels_comb_test[labels_comb_test>=0])


def decode_within_across_tbs(patterns_tb1, patterns_tb2, labels, decoder=SVC(kernel='linear', decision_function_shape='ovo'), train_test_split=0.5, n_loops=10, m_loops=5, chance_loops=5, **args):
    # This will return 2 lists: scores, and scores based on chance (random permutation of labels). size of each = n_loops
    tot_scores_ = [] #temp variable that contains the results of each individual run (n = m_loops)
    tot_scores_chance_ = []
    tot_scores = [] #output that contains the average of each n_loop (n = n_loops)
    tot_scores_chance = []
    for n in range(n_loops): #how many times to run the loop below? final output returns 1 value for each n_loop
        for m in range(m_loops): #run this loop and return the average of all loops
            #define your training and testing datasets (x, y, xT, yT)
            x, y, xT, yT = combine_train_test_patterns_across_tbs(patterns_tb1, patterns_tb2, labels, train_test_split, **args)
            decoder.fit(x, y) #train your decoder using the training data you specified for the diff trial types (x and y)
            scores_for = decoder.score(xT,yT) #test classification accuracy using the held-out data you specified
            decoder.fit(xT,yT) #now, reverse the train-test datasets
            scores_rev = decoder.score(x,y)
            tot_scores_.append(np.mean((scores_for,scores_rev))) #get the mean of forward and reverse directions
            #shuffle labels to get chance scores
            temp = []
            temp_rev = []
            for i in range(chance_loops): #because the permuation is pretty variable, run chance decoding w/ more iterations
                decoder.fit(x,np.random.permutation(y))
                temp.append(decoder.score(xT,yT))
                decoder.fit(xT,np.random.permutation(yT))
                temp_rev.append(decoder.score(x,y))
            scores_chance_for = np.mean(temp)
            scores_chance_rev = np.mean(temp_rev)
            tot_scores_chance_.append(np.mean((scores_chance_for,scores_chance_rev)))
        tot_scores.append(np.mean(tot_scores_))
        tot_scores_chance.append(np.mean(tot_scores_chance_))
    # return mean of forward and reverse decoding, for both real and chance scenarios
    return tot_scores,tot_scores_chance
    
    
def combine_train_test_patterns_individ(patterns, labels, train_test_split=0.5, classes=None, n_cells=None,
                                relabel=None, relabel_test=None,which_10_trials=None):
    #fxn used when want to decode individual animals

    if classes == None:
        classes = range(1,5)

    which_train = {}
    which_test = {}
    which_trains = []
    which_tests = []
    for l in classes:
            if l>=0:
                if which_10_trials == 'first_10':
                    #randomly grab trial indices among first 10 trials of the specified trial type (ie, class)
                    wt = np.random.permutation(np.where(labels==l)[0][:10])
                else:
                    wt = np.random.permutation(np.where(labels==l)[0])
                if train_test_split<1:
                    wT = wt[:int(len(wt)*train_test_split)]
                    which_trains.append(wt[int(len(wt)*train_test_split):])
                    which_tests.append(wT)
                else:
                    which_trains.append(wt)
    which_train = np.concatenate(which_trains)
    if train_test_split<1:
            which_test = np.concatenate(which_tests)
    
    patterns_t = patterns.copy()
    labels_t = labels.copy()
    patterns_t = patterns[which_train]
    labels_t = labels[which_train]
        
    if train_test_split<1:
        patterns_T = patterns.copy()
        labels_T = labels.copy()
        patterns_T = patterns[which_test]
        labels_T = labels[which_test]

    if n_cells is None:
        which_cells = [True]*patterns_t.shape[1]
    else:
        which_cells = np.random.permutation(range(patterns_t.shape[1]))[:n_cells]
     
    patterns_t = patterns_t[:, which_cells]
    patterns_T = patterns_T[:, which_cells] if train_test_split<1 else patterns_T

    return (patterns_t[labels_t>=0], labels_t[labels_t>=0],
            patterns_T[labels_T>=0], labels_T[labels_T>=0])


def decode_within_individ(patterns, labels, decoder=SVC(kernel='linear',decision_function_shape='ovo'),n_loops=10, m_loops=5, chance_loops=5, **args):
    ##This is for decoding individual animals. ONLY DIFF BETWEEN THIS AND DECODE_WITHIN FUNCTION IS THIS USES COMBINE_TRAIN_TEST_PATTERNS_INDIVID.
    #This will return 2 lists: scores, and scores based on chance (random permutation of labels). size of each = n_loops
    tot_scores_ = [] #temp variable that contains the results of each individual run (n = m_loops)
    tot_scores_chance_ = []
    tot_scores = [] #output that contains the average of each n_loop (n = n_loops)
    tot_scores_chance = []
    for n in range(n_loops): #how many times to run the loop below? final output returns 1 value for each n_loop
        for m in range(m_loops): #run this loop and return the average of all loops
            #define your training and testing datasets (x, y, xT, yT)
            x, y, xT, yT = combine_train_test_patterns_individ(patterns, labels, **args) ## THIS IS WHAT DIFFERS FROM DECODE_WITHIN FXN
            decoder.fit(x, y) #train your decoder using the training data you specified for the diff trial types (x and y)
            scores_for = decoder.score(xT,yT) #test classification accuracy using the held-out data you specified
            decoder.fit(xT,yT) #now, reverse the train-test datasets
            scores_rev = decoder.score(x,y)
            tot_scores_.append(np.mean((scores_for,scores_rev))) #get the mean of forward and reverse directions
            #shuffle labels to get chance scores
            temp = []
            temp_rev = []
            for i in range(chance_loops): #because the permuation is pretty variable, run chance decoding w/ more iterations
                decoder.fit(x,np.random.permutation(y))
                temp.append(decoder.score(xT,yT))
                decoder.fit(xT,np.random.permutation(yT))
                temp_rev.append(decoder.score(x,y))
            scores_chance_for = np.mean(temp)
            scores_chance_rev = np.mean(temp_rev)
            tot_scores_chance_.append(np.mean((scores_chance_for,scores_chance_rev)))
        tot_scores.append(np.mean(tot_scores_))
        tot_scores_chance.append(np.mean(tot_scores_chance_))
    # return mean of forward and reverse decoding, for both real and chance scenarios
    return tot_scores,tot_scores_chance

def predict_within_individ(patterns, labels, decoder=SVC(kernel='linear',decision_function_shape='ovo'),chance_loops=5,**args):
    x, y, xT, test_labels = combine_train_test_patterns_individ(patterns, labels, **args)
    decoder.fit(x, y)
    scores = decoder.predict(xT)
    #decoder.fit(xT,test_labels)
    #scores_rev = decoder.predict(x)
    scores_chance = []
    #scores_chance_rev = []
    for i in range(chance_loops):
        decoder.fit(x,np.random.permutation(y))
        scores_chance.append(decoder.predict(xT))
    #    decoder.fit(xT,np.random.permutation(test_labels))
    #    scores_chance_rev.append(decoder.predict(x))
    scores_chance = np.mean(scores_chance)
    #scores_chance_rev = np.mean(scores_chance_rev)
    return scores,scores_chance, test_labels#np.mean((scores,scores_rev)), np.mean((scores_chance,scores_chance_rev)), test_labels


# For Leave-one-out decoding of individual animals
def combine_train_test_patterns_LOO_individ(patterns, labels, classes=None, n_cells=None, which_trials='min'):
    if classes == None:
        classes = [0,1]
    extracted_trial = {}
    extracted_trials = []
    min_trials = np.min([len(np.where(labels==l)[0]) for l in classes])
    if which_trials == 'min':
        for l in classes:
            if l>=0:
                wt = np.random.permutation(np.where(labels==l)[0])
                extracted_trials.append(wt[:min_trials])
    elif which_trials == 'all':
        for l in classes:
            if l>=0:
                wt = np.random.permutation(np.where(labels==l)[0])
                extracted_trials.append(wt)

    extracted_trial = np.concatenate(extracted_trials)
    
    patterns_e = patterns.copy()
    labels_e = labels.copy()
    patterns_e = patterns[extracted_trial]
    labels_e = labels[extracted_trial]
        
    if n_cells == None:
        which_cells = [True]*patterns_e.shape[1]
    else:
        which_cells = np.random.permutation(range(patterns_e.shape[1]))[:n_cells]
     
    patterns_e = patterns_e[:, which_cells]

    return (patterns_e[labels_e>=0], labels_e[labels_e>=0], min_trials)


def predict_LOO_individ(patterns, labels, decoder=SVC(kernel='linear',decision_function_shape='ovo'),cv=LeaveOneOut(), n_loop=10, n_jobs=1, **args):
    scores = []
    scores_chance = []
    ps,ls,min_trials = combine_train_test_patterns_LOO_individ(patterns, labels, **args)
    scores = cross_val_predict(decoder, ps, ls, cv=cv, n_jobs=n_jobs) #setting cv=min_trials should be same as LOO
    #scores_chance = []
    #for i in range(n_loop):
    #    scores_chance.append(cross_val_score(decoder, ps, np.random.permutation(ls), cv=cv, n_jobs=n_jobs))
    #scores_chance = np.r_[scores_chance].flatten()
    return scores,ls#, scores_chance


def CCGP_12vs34(patterns, labels, train_test_split=0.5, decoder=SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000), n_cells=None, which_trials=None, n_loops=10, m_loops=3, chance_loops=3, **args):
    '''Cross-condition generalization performance (CCGP), where labels 1 and 2 share some general property (eg, reward trials), and 3 and 4 another property (eg, CS- trials). Relabel the training data to only compare one label of each type, then test using data of the complementary labels (eg, train: 1vs3, test: 2vs4). Do this for each diff combination of types. High accuracy here indicates that the different labels within a class (eg, 1 and 2) have pop. activity that is more similar to one another than trial types of the other class (eg, 3 and 4).'''
    tot_scores_ = [] #temp variable that contains the results of each individual run (n = m_loops)
    tot_scores_chance_ = []
    tot_scores = [] #output that contains the average of each n_loop (n = n_loops)
    tot_scores_chance = []
    for n in range(n_loops): #how many times to run the loop below? final output returns 1 value for each n_loop
        for m in range(m_loops): #run this loop and return the average of all its loops
            #relabel data so train on one comparison & test on complementary comparison (& vice versa when reversing train/test labels)
            # IMPORTANT: trial types of the same type (eg, 1 and 2 here) MUST share the same label (0 here) in relabel_train and relabel_test
            relabel_train = {1:0, 2:-1, 3:1, 4:-1}
            relabel_test = {1:-1, 2:0, 3:-1, 4:1} #here, we're training 1vs3, and testing 2vs4
            x, y, xT, yT = combine_train_test_patterns(patterns, labels, train_test_split, classes=range(1,5), n_cells=n_cells,
                                               relabel=relabel_train, relabel_test=relabel_test, **args)
            decoder.fit(x, y) #train your decoder using the training data you specified for the diff trial types (x and y)
            scores_for1 = decoder.score(xT,yT) #test classification accuracy using the held-out data you specified
            decoder.fit(xT,yT) #now, reverse the train-test datasets
            scores_rev1 = decoder.score(x,y)
            #shuffle labels to get chance scores
            temp = []
            temp_rev = []
            for i in range(chance_loops): #because the permuation is pretty variable, run chance decoding w/ more iterations
                decoder.fit(x,np.random.permutation(y))
                temp.append(decoder.score(xT,yT))
                decoder.fit(xT,np.random.permutation(yT))
                temp_rev.append(decoder.score(x,y))
            scores_chance_for1 = np.mean(temp)
            scores_chance_rev1 = np.mean(temp_rev)
            
            #now relabel data totrain/test on other complementary pattern
            relabel_train = {1:0, 2:-1, 3:-1, 4:1}
            relabel_test = {1:-1, 2:0, 3:1, 4:-1}
            x, y, xT, yT = combine_train_test_patterns(patterns, labels, train_test_split, classes=range(1, 5), n_cells=n_cells,
                                                   relabel=relabel_train, relabel_test=relabel_test, **args)
            decoder.fit(x, y) #train your decoder using the training data you specified for the diff trial types (x and y)
            scores_for2 = decoder.score(xT,yT) #test classification accuracy using the held-out data you specified
            decoder.fit(xT,yT) #now, reverse the train-test datasets
            scores_rev2 = decoder.score(x,y)
            #shuffle labels to get chance scores
            temp = []
            temp_rev = []
            for i in range(chance_loops): #because the permuation is pretty variable, run chance decoding w/ more iterations
                decoder.fit(x,np.random.permutation(y))
                temp.append(decoder.score(xT,yT))
                decoder.fit(xT,np.random.permutation(yT))
                temp_rev.append(decoder.score(x,y))
            scores_chance_for2 = np.mean(temp)
            scores_chance_rev2 = np.mean(temp_rev)
            
            tot_scores_.append(np.mean((scores_for1,scores_rev1,scores_for2,scores_rev2))) #get the mean of forward and reverse directions
            tot_scores_chance_.append(np.mean((scores_chance_for1,scores_chance_rev1,scores_chance_for2,scores_chance_rev2)))
        #for each n loop, append the mean of the all m loops (which was calculated directly above)
        tot_scores.append(np.mean(tot_scores_))
        tot_scores_chance.append(np.mean(tot_scores_chance_))
    # return mean of forward and reverse decoding of all train/test patterns, for both real and chance scenarios
    return tot_scores,tot_scores_chance


def heatmap(data, row_labels, col_labels, ax=None, cmap='viridis',
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = pl.gca()

    # Plot the heatmap
    im = ax.imshow(data, cmap, origin="lower", **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=7)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    pl.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1])-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0])-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=.35)
    ax.tick_params(axis='both', which='both', bottom=False, left=False,pad=-1)
    
    return im#, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["white", "black"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string == supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def annotate_colormesh(im, data=None, valfmt="{x:.2f}",textcolors=["white", "black"],threshold=None, tot_stims = 4, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data == used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first == used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string == supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(tot_stims):
        for j in range(tot_stims):
            kw.update(color=textcolors[int(im.norm(data[i*tot_stims+j]) > threshold)])
            text = im.axes.text(j+.5, i+.5, valfmt(data[i*tot_stims+j], None), **kw)
            texts.append(text)

    return texts

def compute_cosine_similarity_matrix(pattern_ids, all_patterns):
    corrmat_distr = {}
    #why did we enumerate in the 2 lines below??? Don't see why this is necessary...
    for i, (l, a) in enumerate(zip(pattern_ids, all_patterns)):
        for j, (m, b) in enumerate(zip(pattern_ids, all_patterns)):
            temp = []
            #all_patterns has shape nxmxl, where n is pattern ID (stim and tb), m is # of stim trials/presntations, l is total cells.
            # thus, iterate through each stim presentation below (get cosine similarity of each trial vs each additional trial)
            for ii, aa in enumerate(a):
                for jj, bb in enumerate(b):
                    # skip if same vector or any of the 2 is zero
                    if ii==jj or np.sum(aa)==0 or np.sum(bb)==0: continue
                    # count once if using same data
                    if l==m and jj<ii: continue
                    #reshape array (necessary for cosine similarity function)
                    aaRS = aa.reshape(1,-1)
                    bbRS = bb.reshape(1,-1)
                    temp.append(cosine_similarity(aaRS, bbRS))
            # collect all comparisons for the specific trial and tb combination (l,m)
            corrmat_distr[(l, m)] = temp
    corrmat = np.zeros((len(pattern_ids), len(pattern_ids)))
    for i, p in enumerate(pattern_ids):
        for j, q in enumerate(pattern_ids):
            #take the mean of all comparisons across trials for each trial/tb combo
            corrmat[i][j] = np.mean(corrmat_distr[(p, q)])

    return corrmat_distr, corrmat


from sklearn.covariance import LedoitWolf
from scipy.spatial import distance
def find_MahalDist(patterns1, patterns2,n_loops=10, n_iterations=200):
    
    '''patterns1 = combined activity of responses to event A (array of shape n,m, 
       where n = # of event A trials, m = # of cells. I run this on pseudopopulation, but can also do for individ mice)
       patterns2 = combined activity of responses to event B
       n_iterations = how many times to find MD per loop; result will spit out the mean of all iterations (if the number
       of trials you have is << n cells, you'll want to set this quite high to sample most/all the space of neuron IDs
       across cumulative iterations)
       n_loops = how many times to repeat the above. output of fxn will be a list of size n_loops (with each loop being
       the average of n_iterations)
       
       NOTE: A fundamental issue occurs when the number of neurons is > number of trials. This will very often lead
       to a singular covariance matrix that cannot be inverted. There are several ways to deal with this - a variety of
       covariance estimation techniques exist (eg, Ledoit-Wolf shrinkage or Graphical Lasso estimation).
       Here, I randomly downsample the number of neurons to match the number of trials, repeat many times, and 
       average the result (if an iteration of this still fails, I add regularization to the cov matrix. If even that
       fails, I bail on that iteration and continue with the next).
       An additional stragegy entails applying dimensionality reduction to the population data before running this fxn.
       
       Sometimes, for reasons unclear, the distance for an iteration will be crazy large (> 1000).
       In these cases, I don't save the result of that particular iteration '''
    
    Mahalanobis_Dist=[] #initialize list that will contain result of each loop
    completed = 0  # counter for number of iterations where inverse cov matrix was successfully obtained
    completed2 = 0 # counter for number of iters where inverse cov matrix was successfully obtained after regularization
    failed = 0 # counter for number of iterations where inverse cov matrix failed after both methods
    for loop in range(n_loops):
        mahalanobis_distances = []
        for _ in range(n_iterations):
            # Randomly select a subset of neurons, same size as number of trials
            num_neurons = patterns1.shape[0]
            subsampled_neurons = np.random.choice(patterns1.shape[1],size=num_neurons,replace=False)
        
            # Subsample the data for both stimuli based on the selected neurons
            subsampled_t1_data = patterns1[:, subsampled_neurons]
            subsampled_t2_data = patterns2[:, subsampled_neurons]
        
            ## skip if any of the 2 vectors is zero (can uncomment below if you want to make sure at least 
            ##   one cell is active in your data)
            #if np.sum(subsampled_t1_data)==0 or np.sum(subsampled_t2_data)==0:
            #    print ("y",end=",")
            #    continue
            
            # Combine the subsampled data into a single matrix
            combined_data = np.vstack((subsampled_t1_data, subsampled_t2_data))
        
            # Estimate the covariance matrix using sample covariance estimator
            cov_matrix = np.cov(combined_data, rowvar=False)
        
            # Calculate the inverse covariance matrix
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                #print (".",end=",")
                completed = completed+1
            # if inversion fails, use regularization on covariance matrix (several methods available; using Ledoit-Wolf method here)
            except:
                try:
                    # Estimate the covariance matrix using Ledoit-Wolf shrinkage
                    cov_estimator = LedoitWolf(store_precision=False)
                    cov_estimator.fit(combined_data)
                    cov_matrix = cov_estimator.covariance_
                    inv_cov_matrix = np.linalg.inv(cov_matrix)
                    completed2 = completed2+1
                # if still fails, bail on this iteration
                except:
                    #print (".",end=",")
                    failed=failed+1
                    continue
        
            #average activity of each cell across all presentations of each stimulus
            avg_patterns1= np.mean(patterns1,axis=0)
            avg_patterns2= np.mean(patterns2,axis=0)
            
            # Calculate the Mahalanobis distance
            mahalanobis_distance = distance.mahalanobis(avg_patterns1[subsampled_neurons],
                                                        avg_patterns2[subsampled_neurons], inv_cov_matrix)
            # if dist is crazy large, something is amiss. In this case, dont save the iteration
            #  (There can be several causes of this. Most likely in my case due to poor estimation or nearly singular 
            #   covaiance matrix)
            if mahalanobis_distance > 500:
                print ("***MD = "+str(mahalanobis_distance)+". Too high! Scrapping this iteration****")
            else:
                mahalanobis_distances.append(mahalanobis_distance) #include iteration in result
        
        # Calculate the average Mahalanobis distance over all included iterations (ignore nan values by using np.nanmean)
        # and save the output of each loop
        Mahalanobis_Dist.append(np.nanmean(mahalanobis_distances))
    #print out the # of iterations with each outcome across all loops
    print ("ITERATIONS WHERE:   inv cov found="+str(completed), "   inv cov found w/ regularization="+str(completed2),
           "   falied to get inv cov mat=",str(failed))
    return Mahalanobis_Dist
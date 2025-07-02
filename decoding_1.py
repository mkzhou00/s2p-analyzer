# Decoding 
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import seaborn as sns
import h5py

# Pool data across session and animals
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import seaborn as sns
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.metrics import (
    accuracy_score,
    silhouette_score,
    adjusted_rand_score,
    silhouette_samples,
    confusion_matrix,
)
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split,  cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import (
    ModelDesc,
    EvalEnvironment,
    Term,
    EvalFactor,
    LookupFactor,
    dmatrices,
    INTERCEPT,
)
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import sys
import utils_py3 as ut

from s2p_utils.data_loader import DataLoader
from s2p_utils.processing_utils import (
    correct_overlapping_cells_across_planes,
    correct_timestamps,
    get_cell_only_activity,
    extract_events,
    get_corrected_F,
    normalize_signal,
    extract_Fave_around_events,
    extract_F_around_events,
    reorder_clusters,
)
from plot_utils import (
    plot_raw_licks,
    plot_average_PSTH_around_interest_window,
    plot_individual_cells_activity,
    plot_PC_screenplot,
    plot_PCs,
    make_silhouette_plot,
    plot_activity_clusters,
    plot_cluster_pairs,
    plot_individual_trial_average_activity,
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parses arguments from command line."""
    parser = argparse.ArgumentParser(description="Suite2p result analyzer.")
    
    parser.add_argument(
        "--imaging_system",
        type=str,
        default="INSS",
        help="Which scope imaged at: Bruker or INSS"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        # default="/Users/mzhou/Library/CloudStorage/OneDrive-UCSF/MZ_hpc_prism_M4/d5/",
        default="Z:\\2p\\experiment1\\",
        help="Directory containing all the required files.",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="result",
        help="folder for saving figures and results",
    )
    parser.add_argument(
        "--num_planes",
        type=int,
        default="4",
        help="Number of planes recorded during each session",
    )
    parser.add_argument(
        "--num_flyback",
        type=int,
        default=0,
        help="Number of fly back planes at the end of z stack",
    )
    parser.add_argument(
        "--min_cell_prob",
        type=float,
        default=0.5,
        help="Minimum probability to identify an ROI as a cell.",
    )
    parser.add_argument(
        "--pre_cue_window",
        type=float,
        default=3,
        help="Interested time region before a cue starts, needs to be negative if including time before cue start (Second)",
    )
    parser.add_argument(
        "--post_cue_window",
        type=float,
        default=17,
        help="Interested time region after a cue starts. (Second)",
    )
    parser.add_argument(
        "--delay_to_reward",
        type=int,
        default=3,
        help="Number of seconds from cue onset to reward onset",
    )
    parser.add_argument(
        "--framerate", type=int, default=5, help="Targetted frame rate for analysis across animals across days, 5hz"
    )
    parser.add_argument(
        "--trial_types",
        type=list,
        default=["CS1+", "CS2+", "CS3-"],
        help="Trial types of the experiment",
    )
    parser.add_argument(
        "--neucoeff",
        type=float,
        default=0.7,
        help="neuropil coefficient factor",
    )
    parser.add_argument(
        "--cell_threshold",
        type=int,
        default=10,
        help="threshold percentage difference between F and Fneu to count as a valid cell",
    )
    return parser.parse_args()


def associate_cells_with_intervals(
    interval: list,
    time: float,
    cell_indices_with_activities: "dict[int, set]",
    intervals_with_detected_cells: "dict[int, set]",
    cell_idx: int,
):
    for interval_idx, (start, end) in enumerate(interval):
        if time >= start and time <= end:
            if cell_idx not in cell_indices_with_activities:
                cell_indices_with_activities[cell_idx] = set()
            cell_indices_with_activities[cell_idx].add(interval_idx)

            if interval_idx not in intervals_with_detected_cells:
                intervals_with_detected_cells[interval_idx] = []
            intervals_with_detected_cells[interval_idx].add(cell_idx)
def do_decoding_combined_within(patterns, labels, n_loops=10, ncells=None, cellreg=False, n_steps=10, **args):
     
    scores = np.r_[[do_test_within(patterns, labels, **args) for i in range(n_loops)]]
    
    if ncells is None:
        total_n = np.sum([patterns[ani].shape[1] for ani in animals])
        ncells = np.repeat(np.r_[np.linspace(5, total_n, n_steps).astype(int)], 5)
        
    scores_ncells = np.r_[[do_test_within(patterns, labels, n_cells=n, **args) 
                                        for n in ncells]]
    
    return scores, ncells, scores_ncells


from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

def combine_train_test_patterns(patterns, labels, train_test_split=0.5, classes=None, n_cells=None,
                                relabel=None, relabel_test=None,which_trials='min'):

    if classes is None:
        classes = [0, 1]

    which_train = {}
    which_test = {}
    for ani in patterns.keys():
        which_trains = []
        which_tests = []
        min_trials = np.min([len(np.where(labels[ani]==l)[0]) for l in classes])
        #print       np.min([len(np.where(labels_d1['94_480um']==tc)[0]) for tc in test_classes])
        if which_trials == 'min':
            for l in classes:
                if l>=0:
                    wt = np.random.permutation(np.where(labels[ani]==l)[0][:min_trials])
                    wT = wt[:int(len(wt)*train_test_split)]
                    which_trains.append(wt[int(len(wt)*train_test_split):])
                    which_tests.append(wT)
        elif which_trials == 'all':
            for l in classes:
                if l>=0:
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
    
    patterns_t = patterns.copy()
    labels_t = labels.copy()
    for ani in patterns.keys():
        patterns_t[ani] = patterns[ani][which_train[ani]]
        labels_t[ani] = labels[ani][which_train[ani]]
    patterns_comb_train, labels_comb_ = ut.combine_patterns(patterns_t, labels_t, classes=classes)
        
    if train_test_split<1:
        patterns_T = patterns.copy()
        labels_T = labels.copy()
        for ani in patterns.keys():
            patterns_T[ani] = patterns[ani][which_test[ani]]
            labels_T[ani] = labels[ani][which_test[ani]]
        patterns_comb_test, labels_comb_test_ = ut.combine_patterns(patterns_T, labels_T, classes=classes)
    else:
        patterns_comb_test = None
    
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
    
    
    if n_cells is None:
        which_cells = [True]*patterns_comb_train.shape[1]
    else:
        which_cells = np.random.permutation(range(patterns_comb_train.shape[1]))[:n_cells]
     
    patterns_comb_train = patterns_comb_train[:, which_cells]
    patterns_comb_test = patterns_comb_test[:, which_cells] if train_test_split<1 else patterns_comb_test

    return (patterns_comb_train[labels_comb>=0], labels_comb[labels_comb>=0],
            patterns_comb_test[labels_comb_test>=0], labels_comb_test[labels_comb_test>=0])
        
def do_test_within(patterns, labels, **args):
    x, y, xT, yT = combine_train_test_patterns(patterns, labels, **args)
    decoder.fit(x, y)
    return decoder.score(xT, yT)

def do_test_across(patterns_train, labels_train, patterns_test, labels_test, **args):
    x, y, _, _ = combine_train_test_patterns(patterns_train, labels_train, train_test_split=1, **args)
    xT, yT, _, _ = combine_train_test_patterns(patterns_test, labels_test, train_test_split=1, **args)
    decoder.fit(x, t)
    return decoder.score(xT, yT)

def extract_trials(animal_data, time_slice):
    """
    animal_data: np.array of shape (3, n_trials, n_cells, 20)
    time_slice: slice object (e.g., slice(0, 5) for cue period)
    
    Returns:
        X: (3 * n_trials, n_cells * len(time_slice))
        y: (3 * n_trials,)
    """
    n_CS, n_trials, n_cells, _ = animal_data.shape
    period_data = animal_data[..., time_slice]  # shape: (3, n_trials, n_cells, n_timepoints)
    reshaped = period_data.reshape(n_CS * n_trials, n_cells * period_data.shape[-1])
    labels = np.repeat(np.arange(n_CS), n_trials)  # 0, 1, 2 for CS1+, CS2+, CS3−
    return reshaped, labels
    
def do_decoding(X, y, test_size=0.5 ):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    decoder = LinearSVC()
    decoder.fit(x_train, y_train)
    scores = cross_val_score(decoder, X, y, cv=5)  # 5-fold cross-validation    
    y_pred = decoder.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return scores.mean(), accuracy


def decode_within(patterns,labels,decoder=SVC(kernel='linear',decision_function_shape='ovo'),train_test_split=0.5,n_loops=10,m_loops=3, chance_loops=3, **args):
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

def check_patterns_for_nan(patterns):
    for ani, X in patterns.items():
        if np.isnan(X).any():
            print(f"NaNs in patterns[{ani}]")
        if np.isinf(X).any():
            print(f"Infs in patterns[{ani}]")
        if not np.isfinite(X).all():
            print(f"Non-finite values in patterns[{ani}]")
        if np.max(np.abs(X)) > 1e6:
            print(f"Extremely large values in patterns[{ani}]")

def main():
    # Load data and initialize parameters
    args = parse_args()
    args.learning_stage = "early"

    # Set animals and days for early and late learning
    if args.learning_stage == "early":
        result_dir = "Z:\\2p\\experiment1\\population_data\\early learning\\"
        animal_list = [
            "MZ_CA1_WD_F3",
            "MZ_CA1_WD_M4",
            "MZ_CA1_WD_M5",
            "MZ_CA1_WD_M6",
            "MZ_CA1_WD_M7",
            "MZ_CA1_WD_M8",
            "MZ_CA1_WD_JB_54",
            "MZ_CA1_WD_JB_55",
        ]
        daylist = [1, 1, 1, 2, 1, 1, 3, 3]
    elif args.learning_stage == "late":
        result_dir = "Z:\\2p\\experiment1\\population_data\\late learning\\"
        animal_list = [
            "MZ_CA1_WD_F3",
            "MZ_CA1_WD_M4",
            "MZ_CA1_WD_M5",
            "MZ_CA1_WD_M6",
            "MZ_CA1_WD_M7",
            "MZ_CA1_WD_M8",
            "MZ_CA1_WD_JB_54",
            "MZ_CA1_WD_JB_55"
        ]   
        daylist = [7, 5, 6, 6, 5, 6, 8, 12]

    # animal_ids = []
    
    seed = 42
    n_trial_per_cue = 25
    cue_bins = 3
    trace_bins = [4, 6] # 4 and 5s
    baseline_bine = [0, 2]
    total_bins = slice(0, 20)

    pattern_path = os.path.join(result_dir, "decoding_patterns.pickle")
    label_path = os.path.join(result_dir, "decoding_labels.pickle")
    
    subsampling = np.nan
    if np.isnan(subsampling):
        niteration = 1
    else:
        niteration = 100
    
    if os.path.exists(pattern_path):
        with open(pattern_path, "rb") as f:
            patterns = pickle.load(f)
        with open(label_path, "rb") as f:
            labels = pickle.load(f)    
    
    else:
        patterns = {}
        labels = {}
        for ia, animal in enumerate(animal_list):
            day = daylist[ia]
            animal_dir = os.path.join(args.data_dir, animal, "d"+str(day))
            file_dir = os.path.join(animal_dir, "files")
            
            Fcorr_5hz = np.load(os.path.join(file_dir, "F_5hz.npy"), allow_pickle=True)
            new_im_ts = np.load(os.path.join(file_dir, "timestamps_5hz.npy"), allow_pickle=True)
            
            # # Extract all event time points from new event_df
            num_planes = Fcorr_5hz.shape[0]
            data_loader = DataLoader(animal_dir, num_planes, 0, args.imaging_system)

            event_df = data_loader.get_event_df()  # Arduino
            [licks, CS1, CS2, CS3, sucrose, umami] = extract_events(event_df)
            allCS = [CS1, CS2, CS3]            
            
            # Normalize signal
            Fcorr_norm_down = normalize_signal(
                Fcorr_5hz, num_planes, "median"
            )  # can be z_score, median, robust_z_score

            # # Extract binned Fcorr around pre and post cue window, shape is nCS types x ntrials x nCell x nbins
            Fcorr_around_cue = extract_F_around_events(
                allCS,
                Fcorr_norm_down,
                new_im_ts,
                num_planes,
                args.pre_cue_window,
                args.post_cue_window,
                binsize=1,
                framerate=args.framerate
            )
            
            X, y = extract_trials(Fcorr_around_cue, total_bins)
            patterns[animal] = X
            labels[animal] = y        


        with open(pattern_path, "wb") as f:
            pickle.dump(patterns, f)
        with open(label_path, "wb") as f:
            pickle.dump(labels, f)
    

    time_indices = np.arange(0, 10)  
    accuracy = [[] for x in animal_list]
    accuracy_chance = [[] for x in animal_list]
    
    for t in time_indices:
        sliced_patterns = {}
        for ia, animal in enumerate(animal_list):
            data = patterns[animal]
            n_cells = data.shape[1] // 20
            time_indices = np.arange(n_cells) * 20 + t
            sliced_patterns[animal] = data[:, time_indices]
        
            cs1 = data[0:n_trial_per_cue, time_indices]
            cs2 = data[n_trial_per_cue: n_trial_per_cue*2, time_indices]
            
            performance = []
            performance_chance = []
            
            for iiter in range(niteration):
                performance_temp = []
                performance_chance_temp = []
                if np.isnan(subsampling):
                    cell_idx = np.arange(n_cells)
                else:
                    n_sub = int(n_cells * subsampling)
                    cell_idx = np.random.choice(n_cells, n_sub, replace=False)
                
                for itrial in range(n_trial_per_cue):
                    cs1_train = np.delete(cs1, itrial, axis=0)[:, cell_idx]
                    cs2_train = np.delete(cs2, itrial, axis=0)[:, cell_idx]
                    traindata = np.vstack((cs1_train, cs2_train))
                    trainlabel = np.array([0] * (n_trial_per_cue - 1) + [1] * (n_trial_per_cue - 1))

                    testdata = np.vstack((cs1[itrial, cell_idx], cs2[itrial, cell_idx]))
                    
                    clf = LinearSVC().fit(traindata, trainlabel)
                    testlabel = clf.predict(testdata)
                    performance_temp.append(testlabel==[0,1])
                    
                    np.random.seed(seed)
                    shufflelabel = np.random.permutation(trainlabel)
                    clf_chance = LinearSVC().fit(traindata, shufflelabel)
                    testlabel = clf_chance.predict(testdata)
                    performance_chance_temp.append(testlabel==[0,1])
                    
                performance.append(np.mean(np.concatenate(performance_temp)))                
                performance_chance.append(np.mean(np.concatenate(performance_chance_temp)))
            
            accuracy[ia].append(np.mean(performance))
            accuracy_chance[ia].append(np.mean(performance_chance))
            # scores, chance_scores = decode_within(sliced_patterns, labels, n_loops=5)  
            # accuracy.append(np.mean(scores))
            # chance_results.append(np.mean(chance_scores))
    
    # Mean and sem
    accuracy = np.array(accuracy)              # shape: (n_animals, n_timepoints)
    accuracy_chance = np.array(accuracy_chance)
    
    # 1. Mean and SEM
    mean_real = np.nanmean(accuracy, axis=0)
    sem_real = np.nanstd(accuracy, axis=0) / np.sqrt(accuracy.shape[0])

    mean_chance = np.nanmean(accuracy_chance, axis=0)
    sem_chance = np.nanstd(accuracy_chance, axis=0) / np.sqrt(accuracy_chance.shape[0])

    # 2. Paired t-tests (real vs chance at each timepoint)
    p_values = [stats.ttest_rel(accuracy[:, t], accuracy_chance[:, t]).pvalue for t in range(accuracy.shape[1])]
    significance = ['*' if p < 0.05 else '' for p in p_values]    
    
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(accuracy[0]))
    ax.errorbar(x, mean_real, yerr=sem_real, color='forestgreen', marker='o', linewidth=1, label='Real')
    ax.errorbar(x, mean_chance, yerr=sem_chance, color='gray', marker='o', linestyle='--', linewidth=1, label='Chance')
    
    # Optional for significance scores
    for i, sig in enumerate(significance):
        if sig:
            ax.text(i, max(mean_real[i], mean_chance[i]) + 0.025, sig, ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i-3) for i in x])  # integer tick labels
    ax.set_xlabel('Time from cue onset (s)', fontsize=10)
    ax.set_ylabel('Decoding accuracy', fontsize=10)
    ax.set_ylim([0.4, 0.65])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "CS1vsCS2_decoding.png"), format="png")
    
    
if __name__ == "__main__":
    main()
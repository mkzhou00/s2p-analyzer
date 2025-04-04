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
        "--data_dir",
        type=str,
        required=False,
        # default="/Users/mzhou/Library/CloudStorage/OneDrive-UCSF/MZ_hpc_prism_M4/d5/",
        default="Z:\\2p\\experiment1\\MZ_hpc_prism_M6\\d7\\",
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
        default=4,
        help="Number of planes recorded during each session",
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
        default=-3,
        help="Interested time region before a cue starts (Second)",
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
        "--framerate", type=int, default=5, help="Average frame rate, 5hz"
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

def do_decoding(X, y):
    x_train, x_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)
    decoder = LinearSVC()
    decoder.fit(x_train, y_train)
    scores = cross_val_score(decoder, X, y, cv=5)  # 5-fold cross-validation    
    y_pred = decoder.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return scores.mean(), accuracy


def main():
    args = parse_args()
    args.data_dir = "Z:\\2p\\experiment1\\MZ_hpc_prism_M4\\d15"
    args.num_planes = 2
    # animals = ['MZ_hpc_prism_M6']
    # sessions = ['d6'] 
    data_loader = DataLoader(args.data_dir, args.num_planes)
    file_dir = os.path.join(args.data_dir, "files")
    
    Fcorr = np.load(os.path.join(file_dir, "F.npy"), allow_pickle=True)

    # Load behavioral data and timestamps for images and voltages
    event_df = data_loader.get_event_df()  # Arduino
    voltages = data_loader.get_voltages()  # Computer
    im_ts = data_loader.get_im_ts()  # image time stamps in second
    
    if args.num_planes == 1:
        framerate = np.round(1 / ((im_ts[-1] - im_ts[0]) / len(im_ts))).astype(int)
    else:
        framerate = np.round(1 / ((im_ts[0][-1] - im_ts[0][0]) / len(im_ts[0]))).astype(
            int
        )
    
    # Correct `event_tf` timestamps.
    event_df, new_im_ts = correct_timestamps(event_df, voltages, im_ts, args.num_planes)
    
    # Extract all event time points from new event_df
    [licks, CS1, CS2, CS3, sucrose, milk] = extract_events(event_df)
    allCS = [CS1, CS2, CS3]
    # mintrials = np.min(len(CS1), len(CS2), len(CS3))
    
    # Normalize signal
    Fcorr_norm = normalize_signal(
        Fcorr, args.num_planes, "median"
    )  # can be z_score, median, robust_z_score

    # # Extract average Fcorr around each cue in all cuetypes for each cell, shape is nCS x nCell x nFrames
    Fcorr_cue = extract_F_around_events(
        allCS,
        Fcorr_norm,
        new_im_ts,
        args.num_planes,
        0,
        3,
        framerate=framerate
    )

    F_reshape = Fcorr_cue.reshape(-1, Fcorr_cue.shape[2] * Fcorr_cue.shape[3])
    ntrial = F_reshape.shape[0] // len(allCS)
    F_cs1 = F_reshape[:ntrial]
    F_cs2 = F_reshape[ntrial:ntrial*2]
    F_cs3 = F_reshape[ntrial*2:]
    
    x1 = np.concatenate((F_cs1, F_cs2), axis=0)
    y1 = np.array([0] * ntrial + [1] * ntrial) 

    x2 = np.concatenate((F_cs1, F_cs3), axis=0)
    y2 = np.array([0] * ntrial + [1] * ntrial) 
    
    x3 = np.concatenate((F_cs2, F_cs3), axis=0)
    y3 = np.array([0] * ntrial + [1] * ntrial) 
    

    s1, a1 = do_decoding(x1, y1)
    s2, a2 = do_decoding(x2, y2)
    s3, a3 = do_decoding(x3, y3)    
    print(s3, a3)







    
if __name__ == "__main__":
    main()

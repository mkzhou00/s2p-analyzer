# Decoding 

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
from sklearn.ensemble import RandomForestClassifier

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
    extract_events,
    get_corrected_F,
    normalize_signal,
    extract_Fave_around_events,
    extract_F_around_events,
    reorder_clusters,
    filter_trials_by_minITI,
    extract_patterns_for_decoding,
    get_animal_decoding_dict,
    do_time_resolved_decoding
)
from plot_utils import (
    plot_decoding_accuracy_across_time
)

logger = logging.getLogger(__name__)


def do_time_window_decoding(
    decoded_animals,
    animal_list,
    time_window,  # e.g. [7, 8, 9] for bins 7 to 9
    decoding_pair,
    total_bins_length,
    niteration=100,
    clf=LinearSVC(),
    clf_chance=LinearSVC(),   
    subsampling=np.nan,
    seed=42
):
    """
    Perform decoding using a list of time bins as a window (e.g., [7, 8, 9]) 
    with leave-one-trial-out cross-validation.

    Args:
        decoded_animals (dict): Output from `get_decoded_animals`, per animal.
        animal_list (list): List of animals to decode.
        time_window (list): List of time bin indices to use for decoding.
        decoding_pair (tuple): Trial types to decode (e.g., ("CS1", "CS3")).
        total_bins_length (int): Number of bins per cell.
        niteration (int): Number of iterations (for subsampling).
        subsampling (float or np.nan): Proportion of cells to subsample.
        clf (sklearn classifier): Main classifier.
        clf_chance (sklearn classifier): For chance decoding.
        seed (int): Random seed.

    Returns:
        accuracy (np.ndarray): Shape (n_animals,).
        accuracy_chance (np.ndarray): Same shape, for shuffled labels.
    """
    accuracy = []
    accuracy_chance = []

    for ia, animal in enumerate(animal_list):
        d = decoded_animals[animal]
        data = d["data"]
        local_cell_indices = d["local_cell_indices"]
        n_cells = d["n_cells"]
        n_trial_per_cue = d["n_trial_per_cue"]

        # Combine multiple time bins into one feature vector per cell
        time_indices = np.concatenate([
            local_cell_indices * total_bins_length + t for t in time_window
        ])

        cue_map = {
            "CS1": data[0:n_trial_per_cue, :][:, time_indices],
            "CS2": data[n_trial_per_cue: 2 * n_trial_per_cue, :][:, time_indices],
            "CS3": data[2 * n_trial_per_cue: 3 * n_trial_per_cue, :][:, time_indices],
        }
        cs_a = cue_map[decoding_pair[0]]
        cs_b = cue_map[decoding_pair[1]]

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

            # Each cell contributes len(time_window) features
            feature_idx = np.concatenate([
                cell_idx * len(time_window) + i for i in range(len(time_window))
            ])

            for itrial in range(n_trial_per_cue):
                cs_a_train = np.delete(cs_a, itrial, axis=0)[:, feature_idx]
                cs_b_train = np.delete(cs_b, itrial, axis=0)[:, feature_idx]

                traindata = np.vstack((cs_a_train, cs_b_train))
                trainlabel = np.array([0] * (n_trial_per_cue - 1) + [1] * (n_trial_per_cue - 1))
                testdata = np.vstack((cs_a[itrial, feature_idx], cs_b[itrial, feature_idx]))

                clf.fit(traindata, trainlabel)
                testlabel = clf.predict(testdata)
                performance_temp.append(testlabel == [0, 1])

                np.random.seed(seed)
                shufflelabel = np.random.permutation(trainlabel)
                clf_chance.fit(traindata, shufflelabel)
                testlabel = clf_chance.predict(testdata)
                performance_chance_temp.append(testlabel == [0, 1])

            performance.append(np.mean(np.concatenate(performance_temp)))
            performance_chance.append(np.mean(np.concatenate(performance_chance_temp)))

        accuracy.append(np.mean(performance))
        accuracy_chance.append(np.mean(performance_chance))

    return np.array(accuracy), np.array(accuracy_chance)


    
"""
Start of the file here
steps:
    1) Load each animal's F_5hz data and bin within each trial. Default is 1s bins. So the result patterns for each animal is nCS types x nCell*ntrials*nbins
    2) Decide on a decoder, default is linearSVC()
    3) Decide on decoding for all cells or by cluster from previous clustering on transformed data
    4) Perform leave one trial out decoding across time, training on the rest trials and test on this one; can add more ways to separate trainig/testing data but good for now
    5) Plot accuracy

"""


##--------------------------------------------------------------------------------------------------------
### THINGS NEED TO SET ------------------------------------------------
# 1. Initialize parameters
framerate = 5
trial_types = ["CS1+", "CS2+", "CS3-"]
pre_cue_window = 3
post_cue_window = 17
delay_to_reward = 3
# min_cell_prob = 0.5
# neucoeff = 0.7
# cell_threshold = 10
data_dir = "Z:\\2p\\experiment1"

imaging_system = "INSS"
learning_stage = "late"

# Set animals and days for early and late learning
if learning_stage == "early":
    result_dir = "Z:\\2p\\experiment1\\population_data\\early learning\\d1_first10_trials\\"
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
    daylist = [1, 1, 1, 1, 1, 1, 1, 1]
    subtrials = 'first10'

elif learning_stage == "late":
    result_dir = "Z:\\2p\\experiment1\\population_data\\late learning\\all trials"
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
    # daylist = [7, 5, 6, 6, 5, 6, 8, 12] # before
    # daylist = [7, 5, 5, 5, 6, 6, 8, 12]
    daylist = [7, 5, 6, 6, 6, 6, 8, 12]
    subtrials = 'all'
    
    
# Load time bins for patterns, do all the time for now, can slice later 
seed = 42
total_bins = slice(0, 20)
total_bins_length = total_bins.stop - total_bins.start


##--------------------------------------------------------------------------------------------------------
# 2. Load decoding patterns and labels, if not there, create them
pattern_path = os.path.join(result_dir, "decoding_patterns.pickle")
label_path = os.path.join(result_dir, "decoding_labels.pickle")

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
        animal_dir = os.path.join(data_dir, animal, "d"+str(day))
        file_dir = os.path.join(animal_dir, "files")
        
        Fcorr_5hz = np.load(os.path.join(file_dir, "F_5hz.npy"), allow_pickle=True)
        new_im_ts = np.load(os.path.join(file_dir, "timestamps_5hz.npy"), allow_pickle=True)
        
        # # Extract all event time points from new event_df
        num_planes = Fcorr_5hz.shape[0]
        data_loader = DataLoader(animal_dir, num_planes, 0, imaging_system)

        event_df = pd.read_pickle(os.path.join(file_dir, "event_df.pkl"))  # Arduino
        [licks, CS1, CS2, CS3, sucrose, umami] = extract_events(event_df)
        allCS = filter_trials_by_minITI([CS1, CS2, CS3], post_cue_window)
            
        # Normalize signal
        Fcorr_norm_down = normalize_signal(
            Fcorr_5hz, num_planes, "median"
        )  # can be z_score, median, robust_z_score

        # # Extract binned Fcorr around pre and post cue window for each trial, shape is nCS types x ntrials x nCell x nbins, binsize in ms
        Fcorr_around_cue = extract_F_around_events(
            allCS,
            Fcorr_norm_down,
            new_im_ts,
            num_planes,
            pre_cue_window,
            post_cue_window,
            binsize=1000,
            framerate=framerate
        )
        
        X, y = extract_patterns_for_decoding(Fcorr_around_cue, total_bins)
        patterns[animal] = X
        labels[animal] = y        

    with open(pattern_path, "wb") as f:
        pickle.dump(patterns, f)
    with open(label_path, "wb") as f:
        pickle.dump(labels, f)


##--------------------------------------------------------------------------------------------------------
### THINGS NEED TO SET ------------------------------------------------
# 3. Set decoding parameters
# Choose a decoder, default is linearSVC
clf = LinearSVC()
# clf = SVC(kernel='rbf')
# clf = RandomForestClassifier(n_estimators=100, random_state=seed)
clf_chance = clf        
decoding_pair = ("CS1", "CS2")
testing_pair = decoding_pair
# testing_pair = ("CS2", "CS3")

# set time bins
decoding_time_window = np.arange(0, 20)  # decoding time window
cue_window = np.arange(3,4)
trace_window = np.arange(4,6)
postUS_window_0 = np.arange(6,9)
postUS_window_1 = np.arange(9,12)
# postUS_window_2 = np.arange(12,15)

accuracy = [[] for x in animal_list]
accuracy_chance = [[] for x in animal_list]
subsampling = np.nan
if np.isnan(subsampling):
    niteration = 5
else:
    niteration = 10

# decided whether decode by clusters
cluster_labels = np.load(os.path.join(result_dir, "clusterlabels.npy")) # each cell's cluster label
populationdata = np.load(os.path.join(result_dir, "populationdata.npy"))
animal_id = np.load(os.path.join(result_dir, "animal_id.npy")) # each cell's animal ID
decode_by_cluster = False
selected_clusters = np.arange(0, 9) # zero indexed, usually from 0 to 8

for cluster in selected_clusters:
    ##--------------------------------------------------------------------------------------------------------
    # 4. Do decoding
    # Get each animal's decoding data with cell indices and cluster ID if needed
    decoded_animals = get_animal_decoding_dict(
        animal_list=animal_list,
        patterns=patterns,
        trial_types=trial_types,
        total_bins_length=total_bins_length,
        decode_by_cluster=decode_by_cluster,
        cluster_labels=cluster_labels,
        selected_clusters=cluster,
        animal_id=animal_id,
    )

    # run decoding across time 
    accuracy, accuracy_chance = do_time_resolved_decoding(
        decoded_animals,
        animal_list,
        decoding_time_window,
        decoding_pair,
        total_bins_length,
        subtrials=subtrials,
        niteration=niteration,
        clf=clf,
        clf_chance=clf_chance,   
        subsampling=subsampling,
        seed=seed,
        testing_pair=testing_pair)

    # accuracy_fixed_time, accuracy_chance_fixed_time = do_time_window_decoding(
    #     decoded_animals,
    #     animal_list,
    #     time_window=,
    #     decoding_pair,
    #     niteration,
    #     clf,
    #     clf_chance,
    #     subsampling,
    #     seed
    # )

    ##--------------------------------------------------------------------------------------------------------
    # 5. Plotting
    fig_accuracy_across_time = plot_decoding_accuracy_across_time(accuracy, accuracy_chance)
    decoder_name = clf.__class__.__name__
    if decode_by_cluster:
        plot_filename = f"{testing_pair[0]}vs{testing_pair[1]}_decoding_{decoder_name}_cluster{cluster+1}.png"
        fig_accuracy_across_time.savefig(os.path.join(result_dir, plot_filename), format="png")
    else:
        plot_filename = f"{testing_pair[0]}vs{testing_pair[1]}_decoding_{decoder_name}.png"    
        fig_accuracy_across_time.savefig(os.path.join(result_dir, plot_filename), format="png")
        break
    plt.close(fig_accuracy_across_time)


    # ## Optional: plot the mean PSTH for this cluster for sanity check
    # colors = {'CS1':(0, 0.5, 1), 'CS2': (1, 0.5, 0), 'CS3':(0.8, 0.8, 0.8)}
    # time_axis = np.arange(total_bins_length) - pre_cue_window

    # # Initialize accumulators for each CS type
    # psth_by_cue = {
    #     'CS1': [],
    #     'CS2': [],
    #     'CS3': []
    # }

    # # Aggregate PSTHs across animals
    # for animal in decoded_animals:
    #     d = decoded_animals[animal]
    #     data = d["data"]  # (n_trials, n_features)
    #     local_indices = d["local_cell_indices"]
    #     n_trial_per_cue = d["n_trial_per_cue"]

    #     n_cells = data.shape[1] // total_bins_length
    #     reshaped_data = data.reshape(data.shape[0], n_cells, total_bins_length)
    #     selected_data = reshaped_data[:, local_indices, :]  # (n_trials, selected_cells, time)

    #     # Split by cue and average over trials and cells
    #     cue_map = {
    #         'CS1': selected_data[0:n_trial_per_cue],
    #         'CS2': selected_data[n_trial_per_cue:2*n_trial_per_cue],
    #         'CS3': selected_data[2*n_trial_per_cue:3*n_trial_per_cue],
    #     }

    #     for cue_name, trials in cue_map.items():
    #         psth = trials.mean(axis=(0, 1))  # mean across trials and cells
    #         psth_by_cue[cue_name].append(psth)

    # # # Compute grand average PSTH for each cue
    # plt.figure(figsize=(4, 4))
    # for cue_name in ['CS1', 'CS2', 'CS3']:
    #     all_psths = np.stack(psth_by_cue[cue_name])  # shape: (n_animals, time)
    #     mean_psth = all_psths.mean(axis=0)
    #     sem_psth = all_psths.std(axis=0) / np.sqrt(all_psths.shape[0])  # SEM

    #     plt.plot(time_axis, mean_psth, label=cue_name, color=colors[cue_name])

    # plt.title(f"Average PSTH Across Animals (Cluster {cluster+1})")
    # plt.xlabel("Time (bins)")
    # plt.xticks(time_axis)
    # plt.ylabel("Mean activity")
    # plt.legend()
    # plt.tight_layout()
    # plt.close()
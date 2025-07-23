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
from sklearn.svm import SVC
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
    correct_overlapping_cells_across_planes,
    correct_timestamps,
    get_cell_only_activity,
    extract_events,
    get_corrected_F,
    normalize_signal,
    extract_Fave_around_events,
    extract_F_around_events,
    reorder_clusters,
    filter_trials_by_minITI
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


def get_decoded_animals(
    animal_list,
    patterns,
    trial_types,
    total_bins_length,
    decode_by_cluster=False,
    cluster_labels=None,
    selected_clusters=None,
    animal_id=None,
):
    """
    Prepares a dictionary of decoded animal data for decoding or PSTH plotting.

    Returns:
        decoded_animals (dict): keys are animal IDs, values contain data, local cell indices, and metadata.
    """
    if decode_by_cluster:
        animal_cell_starts = {}
        start_idx = 0
        for a in animal_list:
            n_cells = np.sum(np.char.find(animal_id.astype(str), a) >= 0)
            animal_cell_starts[a] = start_idx
            start_idx += n_cells
        
    decoded_animals = {}

    for animal in animal_list:
        data = patterns[animal]
        n_trial_per_cue = data.shape[0] // len(trial_types)
        total_cells = data.shape[1] // total_bins_length

        assert data.shape[1] % total_bins_length == 0, \
            f"Unexpected number of features in {animal}: {data.shape[1]}"

        if decode_by_cluster:
            # Get global indices of matching cells
            global_mask = (
                (np.char.find(animal_id.astype(str), animal) >= 0) &
                np.isin(cluster_labels, selected_clusters)
            )
            global_indices = np.where(global_mask)[0]
            local_start = animal_cell_starts[animal]
            local_indices = global_indices - local_start

            if len(local_indices) == 0:
                print(f"[{animal}] No cells in selected clusters {selected_clusters}. Skipping.")
                continue
        else:
            local_indices = np.arange(total_cells)

        decoded_animals[animal] = {
            "data": data,
            "local_cell_indices": local_indices,
            "n_trial_per_cue": n_trial_per_cue,
            "n_cells": len(local_indices),
        }

    return decoded_animals


def do_time_resolved_decoding(
    decoded_animals,
    animal_list,
    decoding_time_window,
    decoding_pair,
    total_bins_length,
    niteration=100,
    clf=LinearSVC(),
    clf_chance=LinearSVC(),   
    subsampling=np.nan,
    seed=42
):
    """
    Perform time-resolved decoding with leave-one-trial-out cross-validation.

    Args:
        decoded_animals (dict): Output from `get_decoded_animals`, per animal.
        animal_list (list): List of animals to decode.
        decoding_time_window (array): Time bins to decode at.
        decoding_pair (tuple): Trial types to decode (e.g., ("CS1", "CS3")).
        total_bins_length (int): Number of bins per cell.
        niteration (int): Number of iterations (for subsampling).
        subsampling (float or np.nan): Proportion of cells to subsample.
        clf (sklearn classifier): Main classifier.
        clf_chance (sklearn classifier): For chance decoding.
        seed (int): Random seed.

    Returns:
        accuracy (np.ndarray): Shape (n_animals, n_timepoints).
        accuracy_chance (np.ndarray): Same shape, for shuffled labels.
    """
    accuracy = [[] for _ in animal_list]
    accuracy_chance = [[] for _ in animal_list]

    for t in decoding_time_window:
        for ia, animal in enumerate(animal_list):

            d = decoded_animals[animal]
            data = d["data"]
            local_cell_indices = d["local_cell_indices"]
            n_cells = d["n_cells"]
            n_trial_per_cue = d["n_trial_per_cue"]

            time_indices = local_cell_indices * total_bins_length + t

            cue_map = {
                "CS1": data[0:n_trial_per_cue, time_indices],
                "CS2": data[n_trial_per_cue: 2 * n_trial_per_cue, time_indices],
                "CS3": data[2 * n_trial_per_cue: 3 * n_trial_per_cue, time_indices],
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

                for itrial in range(n_trial_per_cue):
                    cs_a_train = np.delete(cs_a, itrial, axis=0)[:, cell_idx]
                    cs_b_train = np.delete(cs_b, itrial, axis=0)[:, cell_idx]

                    traindata = np.vstack((cs_a_train, cs_b_train))
                    trainlabel = np.array([0] * (n_trial_per_cue - 1) + [1] * (n_trial_per_cue - 1))
                    testdata = np.vstack((cs_a[itrial, cell_idx], cs_b[itrial, cell_idx]))

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

            accuracy[ia].append(np.mean(performance))
            accuracy_chance[ia].append(np.mean(performance_chance))
            # scores, chance_scores = decode_within(sliced_patterns, labels, n_loops=5)  
            # accuracy.append(np.mean(scores))
            # chance_results.append(np.mean(chance_scores))

    return np.array(accuracy), np.array(accuracy_chance)

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
learning_stage = "early"

# Set animals and days for early and late learning
if learning_stage == "early":
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

elif learning_stage == "late":
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

# Load time bin IDs 
seed = 42
# n_trial_per_cue = 25
cue_bins = 3
trace_bins = [4, 6] # 4 and 5s
baseline_bine = [0, 2]
total_bins = slice(0, 20)
total_bins_length = total_bins.stop - total_bins.start

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

        event_df = data_loader.get_event_df()  # Arduino
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
        
        X, y = extract_trials(Fcorr_around_cue, total_bins)
        patterns[animal] = X
        labels[animal] = y        


    with open(pattern_path, "wb") as f:
        pickle.dump(patterns, f)
    with open(label_path, "wb") as f:
        pickle.dump(labels, f)


### THINGS NEED TO SET ------------------------------------------------
# 3. Set decoding parameters
# Choose a decoder, default is linearSVC
clf = LinearSVC()
# clf = SVC(kernel='rbf')
# clf = RandomForestClassifier(n_estimators=100, random_state=seed)
clf_chance = clf        
decoding_pair = ("CS1", "CS2")

# set time bins
decoding_time_window = np.arange(0, 20)  # decoding time window
accuracy = [[] for x in animal_list]
accuracy_chance = [[] for x in animal_list]
subsampling = np.nan
if np.isnan(subsampling):
    niteration = 1
else:
    niteration = 100

# decided whether decode by clusters
decode_by_cluster = True
if decode_by_cluster is True:
    cluster_labels = np.load(os.path.join(result_dir, "clusterlabels.npy")) # each cell's cluster label
    populationdata = np.load(os.path.join(result_dir, "populationdata.npy"))
    animal_id = np.load(os.path.join(result_dir, "animal_id.npy")) # each cell's animal ID
    selected_clusters = 0 # zero indexed, usually from 0 to 8


# 4. Do decoding
# Get each animal's decoding data with cell indices and cluster ID if needed
decoded_animals = get_decoded_animals(
    animal_list=animal_list,
    patterns=patterns,
    trial_types=trial_types,
    total_bins_length=total_bins_length,
    decode_by_cluster=True,
    cluster_labels=cluster_labels,
    selected_clusters=selected_clusters,
    animal_id=animal_id,
)

# actually run decoding  
accuracy, accuracy_chance = do_time_resolved_decoding(
    decoded_animals,
    animal_list,
    decoding_time_window,
    decoding_pair,
    total_bins_length,
    niteration=niteration,
    clf=clf,
    clf_chance=clf_chance,   
    subsampling=np.nan,
    seed=seed)

# 5. Calculate Mean and sem from the accuracy
mean_real = np.nanmean(accuracy, axis=0)
sem_real = np.nanstd(accuracy, axis=0) / np.sqrt(accuracy.shape[0])
mean_chance = np.nanmean(accuracy_chance, axis=0)
sem_chance = np.nanstd(accuracy_chance, axis=0) / np.sqrt(accuracy_chance.shape[0])

# 6. Do paired t-tests real vs chance at each timepoint
p_values = [stats.ttest_rel(accuracy[:, t], accuracy_chance[:, t]).pvalue for t in range(accuracy.shape[1])]
significance = ['*' if p < 0.05 else '' for p in p_values]    

# 7. Plotting of accuracies across time
fig, ax = plt.subplots(figsize=(5, 3))
x = np.arange(len(accuracy[0]))
ax.errorbar(x, mean_real, yerr=sem_real, color='forestgreen', marker='o', linewidth=1, label='Real')
ax.errorbar(x, mean_chance, yerr=sem_chance, color='gray', marker='o', linestyle='--', linewidth=1, label='Chance')

# Optional for significance scores
for i, sig in enumerate(significance):
    if sig:
        ax.text(i, max(mean_real[i], mean_chance[i]) + 0.025, sig, ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([str(i-3) for i in x])  # integer tick labels
ax.set_xlabel('Time from cue onset (s)', fontsize=10)
ax.set_ylabel('Decoding accuracy', fontsize=10)
# ax.set_ylim([0.4, 0.65])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2), frameon=False)
fig.tight_layout()

# Save file
decoder_name = clf.__class__.__name__
title_str = f"{decoding_pair[0]} vs {decoding_pair[1]} decoding"
# ax.set_title(title_str, fontsize=12)
plot_filename = f"{decoding_pair[0]}vs{decoding_pair[1]}_decoding_{decoder_name}_cluster{selected_clusters+1}.png"
fig.savefig(os.path.join(result_dir, plot_filename), format="png")


## 8. Optional: plot the mean PSTH for this cluster for sanity check
colors = {'CS1':(0, 0.5, 1), 'CS2': (1, 0.5, 0), 'CS3':(0.8, 0.8, 0.8)}
time_axis = np.arange(total_bins_length) - pre_cue_window

# Initialize accumulators for each CS type
psth_by_cue = {
    'CS1': [],
    'CS2': [],
    'CS3': []
}

# Aggregate PSTHs across animals
for animal in decoded_animals:
    d = decoded_animals[animal]
    data = d["data"]  # (n_trials, n_features)
    local_indices = d["local_cell_indices"]
    n_trial_per_cue = d["n_trial_per_cue"]

    n_cells = data.shape[1] // total_bins_length
    reshaped_data = data.reshape(data.shape[0], n_cells, total_bins_length)
    selected_data = reshaped_data[:, local_indices, :]  # (n_trials, selected_cells, time)

    # Split by cue and average over trials and cells
    cue_map = {
        'CS1': selected_data[0:n_trial_per_cue],
        'CS2': selected_data[n_trial_per_cue:2*n_trial_per_cue],
        'CS3': selected_data[2*n_trial_per_cue:3*n_trial_per_cue],
    }

    for cue_name, trials in cue_map.items():
        psth = trials.mean(axis=(0, 1))  # mean across trials and cells
        psth_by_cue[cue_name].append(psth)

# Compute grand average PSTH for each cue
plt.figure(figsize=(4, 4))
for cue_name in ['CS1', 'CS2', 'CS3']:
    all_psths = np.stack(psth_by_cue[cue_name])  # shape: (n_animals, time)
    mean_psth = all_psths.mean(axis=0)
    sem_psth = all_psths.std(axis=0) / np.sqrt(all_psths.shape[0])  # SEM

    plt.plot(time_axis, mean_psth, label=cue_name, color=colors[cue_name])

plt.title(f"Average PSTH Across Animals (Cluster {selected_clusters}+1)")
plt.xlabel("Time (bins)")
plt.xticks(time_axis)
plt.ylabel("Mean activity")
plt.legend()
plt.tight_layout()
plt.show()
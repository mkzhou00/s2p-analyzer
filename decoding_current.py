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


# 1. Initialize parameters
framerate = 5
trial_types = ["CS1+", "CS2+", "CS3-"]
pre_cue_window = 3
post_cue_window = 17
delay_to_reward = 3
min_cell_prob = 0.5
neucoeff = 0.7
cell_threshold = 10
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
    
# 3. Load time bin IDs 
seed = 42
n_trial_per_cue = 25
cue_bins = 3
trace_bins = [4, 6] # 4 and 5s
baseline_bine = [0, 2]
total_bins = slice(0, 20)

# 4. Load decoding patterns and labels, if not there, create them
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
            pre_cue_window,
            post_cue_window,
            binsize=1,
            framerate=framerate
        )
        
        X, y = extract_trials(Fcorr_around_cue, total_bins)
        patterns[animal] = X
        labels[animal] = y        


    with open(pattern_path, "wb") as f:
        pickle.dump(patterns, f)
    with open(label_path, "wb") as f:
        pickle.dump(labels, f)

# 4. Decoding parameters
# Choose a decoder
# clf = LinearSVC()
# clf = SVC(kernel='rbf')
clf = RandomForestClassifier(n_estimators=100, random_state=seed)
clf_chance = clf        
decoding_pair = ("CS1", "CS3")

decoding_time_window = np.arange(0, 20)  # decoding time window
accuracy = [[] for x in animal_list]
accuracy_chance = [[] for x in animal_list]
subsampling = np.nan
if np.isnan(subsampling):
    niteration = 1
else:
    niteration = 100

for t in decoding_time_window:
    sliced_patterns = {}
    for ia, animal in enumerate(animal_list):
        data = patterns[animal]
        
        assert data.shape[1] % 20 == 0, f"Unexpected number of features: {data.shape[1]}"
        n_cells = data.shape[1] // 20
        time_indices = np.arange(n_cells) * 20 + t
        sliced_patterns[animal] = data[:, time_indices]

        # Setting training data
        cue_map = cue_map = {
        "CS1": data[0:n_trial_per_cue, time_indices],
        "CS2": data[n_trial_per_cue: 2*n_trial_per_cue, time_indices],
        "CS3": data[2*n_trial_per_cue: 3*n_trial_per_cue, time_indices],
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
                cs_a_train = np.delete(cs_a, itrial, axis=0)[:, cell_idx]        # Leave one trial out for testing for both trial types
                cs_b_train = np.delete(cs_b, itrial, axis=0)[:, cell_idx]

                traindata = np.vstack((cs_a_train, cs_b_train))
                trainlabel = np.array([0] * (n_trial_per_cue - 1) + [1] * (n_trial_per_cue - 1))
                testdata = np.vstack((cs_a[itrial, cell_idx], cs_b[itrial, cell_idx]))  
                              
                clf.fit(traindata, trainlabel)
                testlabel = clf.predict(testdata)
                performance_temp.append(testlabel==[0,1])
                
                np.random.seed(seed)
                shufflelabel = np.random.permutation(trainlabel)
                clf_chance.fit(traindata, shufflelabel)
                testlabel = clf_chance.predict(testdata)
                performance_chance_temp.append(testlabel==[0,1])
                
            performance.append(np.mean(np.concatenate(performance_temp)))                
            performance_chance.append(np.mean(np.concatenate(performance_chance_temp)))
        
        accuracy[ia].append(np.mean(performance))
        accuracy_chance[ia].append(np.mean(performance_chance))
        # scores, chance_scores = decode_within(sliced_patterns, labels, n_loops=5)  
        # accuracy.append(np.mean(scores))
        # chance_results.append(np.mean(chance_scores))

# 5. Calculate Mean and sem
accuracy = np.array(accuracy)              # shape: (n_animals, n_timepoints)
accuracy_chance = np.array(accuracy_chance)

mean_real = np.nanmean(accuracy, axis=0)
sem_real = np.nanstd(accuracy, axis=0) / np.sqrt(accuracy.shape[0])
mean_chance = np.nanmean(accuracy_chance, axis=0)
sem_chance = np.nanstd(accuracy_chance, axis=0) / np.sqrt(accuracy_chance.shape[0])

# 6. Paired t-tests (real vs chance at each timepoint)
p_values = [stats.ttest_rel(accuracy[:, t], accuracy_chance[:, t]).pvalue for t in range(accuracy.shape[1])]
significance = ['*' if p < 0.05 else '' for p in p_values]    

# 7. Plotting
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

decoder_name = clf.__class__.__name__
title_str = f"{decoding_pair[0]} vs {decoding_pair[1]} decoding"
# ax.set_title(title_str, fontsize=12)
plot_filename = f"{decoding_pair[0]}vs{decoding_pair[1]}_decoding_{decoder_name}.png"
fig.savefig(os.path.join(result_dir, plot_filename), format="png")
# Pool data across session and animals
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
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
)
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.manifold import spectral_embedding   # same code SC uses
from sklearn.cluster._spectral import discretize 
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
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
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC, LinearSVC

from s2p_utils.data_loader import load_population_data
from s2p_utils.processing_utils import (
    reorder_clusters,
    build_knn,
    get_initial_cluster_labels,
    get_filtered_rois_per_animal_plane,
    load_population_data_filtered
)
from plot_utils import (
    tsplot,
    standardize_plot_graphics,
    plot_PC_screenplot,
    plot_PCs,
    make_silhouette_plot,
    plot_activity_clusters,
    plot_cluster_pairs,
    plot_individual_trial_average_activity,
)

logger = logging.getLogger(__name__)


# Initialize parameters
data_dir = "Z:\\2p\\experiment1\\"
pre_cue_window = 3
post_cue_window = 17
delay_to_reward = 3
min_cell_prob = 0.5
neucoeff = 0.7
framerate = 5
target_frames = 300 # Should be the total time window * framerate * nCS, so 300 for the current settings
trial_types = ["CS1+", "CS2+", "CS3-"]
learning_stage = 'late'


# Set animals and days for early and late learning
if learning_stage == "early":
    result_dir = "Z:\\2p\\experiment1\\population_data\\early learning\\"
    animal_list = [
        "MZ_CA1_WD_F3",
        "MZ_CA1_WD_M5",
        "MZ_CA1_WD_M6",
        "MZ_CA1_WD_M7",
        "MZ_CA1_WD_M8",
        "MZ_CA1_WD_JB_54",
        "MZ_CA1_WD_JB_55"
    ]
    day_list = np.ones(7, dtype=int)
    subtrials = 'first10'

elif learning_stage == "intermediate":
    result_dir = "Z:\\2p\\experiment1\\population_data\\intermediate learning\\"
    animal_list = [
        "MZ_CA1_WD_F3",
        # "MZ_CA1_WD_M4/d1",
        "MZ_CA1_WD_M5",
        "MZ_CA1_WD_M6",
        "MZ_CA1_WD_M7",
        "MZ_CA1_WD_M8",
        "MZ_CA1_WD_JB_54",
        "MZ_CA1_WD_JB_55"
    ]
    day_list = [3, 3, 4, 4, 4, 4, 7]
    subtrials = 'all'
      
elif learning_stage == "late":
    result_dir = "Z:\\2p\\experiment1\\population_data\\late learning\\"
    animal_list = [
        "MZ_CA1_WD_F3",
        # "MZ_CA1_WD_M4/d5",
        "MZ_CA1_WD_M5",
        "MZ_CA1_WD_M6",
        "MZ_CA1_WD_M7",
        "MZ_CA1_WD_M8",
        "MZ_CA1_WD_JB_54",
        "MZ_CA1_WD_JB_55"
    ]   
    day_list = [7, 6, 5, 6, 6, 8, 12]
    subtrials = 'all'
    
    
# For plotting
window_size = 100
frames_to_reward = delay_to_reward * framerate
pre_window_size = pre_cue_window * framerate

# %debug
# Load and concatenate population data across animals
populationdata, animal_id = load_population_data(animal_list, day_list, data_dir, result_dir, target_frames, pre_cue_window, framerate, subtrials=subtrials)

# optional for training on certain trial types
cs1_data = populationdata[:, :window_size]
cs2_data = populationdata[:, window_size:2*window_size]
cs3_data = populationdata[:, 2*window_size:]
# train_data = np.hstack([cs1_data, cs3_data])
# trial_types = ['CS1+','CS3-']

train_data = populationdata
trial_types = ['CS1+', 'CS2+', 'CS3-']

# Define cache file paths before running PCA
pca_path = os.path.join(result_dir, "pca_model.pickle")
transformed_path = os.path.join(result_dir, "transformed_data.npy")

# Load cluster labels if exist
if os.path.exists(os.path.join(result_dir, "clusterlabels.npy")):
    newlabels = np.load(os.path.join(result_dir, "clusterlabels.npy"))   
    uniquelabels = list(set(newlabels))
    assert len(newlabels)==train_data.shape[0], "Cluster labels length does not match data points."
    print("Loaded existing cluster labels, skip clustering step.")
    
    
# Load ROI ID for each animal on the previous training day "late"
trained_day_list = [7, 6, 5, 6, 6, 8, 12]
target_day_list = [[1, 2, 3, 5, 10],
                   [1, 2, 3, 5, 10],
                   [1, 2, 3, 6, 9],
                   [1, 2, 3, 5, 9],
                   [1, 2, 3, 5, 10],
                   [1, 2, 3, 5, 7, 10],
                   [1, 2, 3, 9, 14]]
skip_map = {"MZ_CA1_WD_F3": 2} # (plane_index, day_index) to skip missing data
filtered_ROIs_across_sessions, filtered_labels = get_filtered_rois_per_animal_plane(animal_list, trained_day_list, data_dir, newlabels, target_day_list, skip_map)


test_day_list = np.ones(7, dtype=int) # input signle day for each animal
filtered_d1, animal_list, foundcell_idx, foundcell_flags, new_labels_test_day = load_population_data_filtered(
    animal_list, 
    test_day_list, 
    data_dir, 
    result_dir, 
    target_frames, 
    pre_cue_window, 
    framerate, 
    filtered_ROIs_across_sessions)

filtered_ROIs_across_sessions['MZ_CA1_WD_F3'][0]


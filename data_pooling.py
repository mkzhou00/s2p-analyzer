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
    extract_interest_time_intervals,
    extract_imaging_ts_around_events,
    normalize_signal,
    extract_Fave_around_events,
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
        default="4",
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
        default=3,
        help="Interested time region before a cue starts. (Second)",
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



def main():
    args = parse_args()
    args.data_dir = "Z:\\2p\\experiment1\\MZ_hpc_prism_M6\\d7"
    # animals = ['MZ_hpc_prism_M6']
    # sessions = ['d6'] 
    data_loader = DataLoader(args.data_dir, args.num_planes)
    file_dir = os.path.join(args.data_dir, "files")
    
    decoder = LinearSVC()
    
    Fcorr = np.load(os.path.join(file_dir, "F.npy"), allow_pickle=True)

    # Load behavioral data and timestamps for images and voltages
    event_df = data_loader.get_event_df()  # Arduino
    voltages = data_loader.get_voltages()  # Computer
    im_ts = data_loader.get_im_ts()  # image time stamps in second
    # Correct `event_tf` timestamps.
    event_df, new_im_ts = correct_timestamps(event_df, voltages, im_ts, args.num_planes)

    # Extract all event time points from new event_df
    [licks, CS1, CS2, CS3, sucrose, milk] = extract_events(event_df)
    allCS = [CS1, CS2, CS3]
    
    # Extract time around each cue and sorted by CS type, shape is numCS --> len trials
    interest_intervals = extract_interest_time_intervals(
        allCS, args.pre_cue_window, args.post_cue_window
    )
    # Extract image time points around each cue and sorted by CS type and plane, shape is plane --> numCS --> len trials
    im_idx_around_cue = extract_imaging_ts_around_events(
        allCS, new_im_ts, args.num_planes, interest_intervals
    )

    # Normalize signal
    Fcorr_norm = normalize_signal(
        Fcorr, args.num_planes, "median"
    )  # can be z_score, median, robust_z_score

    # # Extract average Fcorr around each cue in all cuetypes for each cell
    Fcorr_cue = extract_Fave_around_events(
        allCS,
        Fcorr_norm,
        im_idx_around_cue,
        args.num_planes,
        0,
        1,
    )
    # reshape the signal to 
    Fcorr_cue = Fcorr_cue.transpose(1, 2, 0).reshape(
        Fcorr_cue.shape[1], -1, order="F"
    ) 
    
     # # Extract average Fcorr around each cue in all cuetypes for each cell
    Fcorr_trace = extract_Fave_around_events(
        allCS,
        Fcorr_norm,
        im_idx_around_cue,
        args.num_planes,
        -1,
        3,
    )
    # reshape the signal to 
    Fcorr_trace = Fcorr_trace.transpose(1, 2, 0).reshape(
        Fcorr_trace.shape[1], -1, order="F"
    )    
    
    
    
    
    
    
if __name__ == "__main__":
    main()

"""
READ BEFORE START:
    FOR BRUKER ANALYSIS:
        Data format:
        - In each animal's file, create a subfoler for each day (called d1, d2 etc)
        - In each day's subfolder, have a folder called "files" containing all the recorded time points (xml), voltage recording (excel) and behavior (mat)
        - suite2p folder is created automatically with suite2p, needs to have mat file for all information
        - can have the reference folder in this subfolder as well
        - running this file will make a result folder in this folder
    FOR INSS ANALYSIS:
        Data format:
        - In each animal's file, create a subfoler for each day (called d1, d2 etc)
        - In each day's folder, should have the tiff image containing all the frames, the behavioral data as mat file, suite2p folder containing the preproccessed 
        files, and 
"""

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
)
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
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

from s2p_utils.data_loader import DataLoader
from s2p_utils.processing_utils import (
    correct_overlapping_cells_across_planes,
    correct_timestamps,
    get_cell_only_activity,
    extract_events,
    get_corrected_F,
    extract_interest_time_intervals,
    extract_imaging_ts_around_events,
    extract_cues_from_voltages,
    normalize_signal,
    extract_Fave_around_events,
    reorder_clusters,
    downsample_data,
    extract_F_around_events,
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


## ----------------------------------------------------------------------------
# 1. Initialize parameters
framerate = 5 # target framerate for all animals
trial_types = ["CS1+", "CS2+", "CS3-"]
pre_cue_window = 3
post_cue_window = 17
delay_to_reward = 3
neucoeff = 0.7
main_folder = "Z:\\2p\\experiment1"

# Animal specific parameters
animal = "MZ_CA1_WD_F3"
days = [2]
num_planes_list = np.ones(12, dtype=int)*4
num_flyback_list = np.ones(12, dtype=int)*0

if animal == "MZ_CA1_WD_F3":
    imaging_system = "Bruker"
else:
    imaging_system = "INSS"


## ----------------------------------------------------------------------------
# 2. Loading each day's data
for id, day in enumerate(days):
    data_dir = os.path.join(main_folder, animal, "d"+str(day))
    num_planes = num_planes_list[id]
    data_loader = DataLoader(data_dir, num_planes, num_flyback_list[id], imaging_system)

    # Make a result folder if if didn't exist
    result_dir = os.path.join(data_dir, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    file_dir = os.path.join(data_dir, "files")
    # Check if files folder exist
    assert os.path.exists(file_dir), "Forgot to make a files folder :/"

    if os.path.exists(os.path.join(file_dir, "F.npy")):
        Fcorr = np.load(os.path.join(file_dir, "F.npy"), allow_pickle=True)
    else:
        # Load all necessary variables.
        F = data_loader.get_F()
        Fneu = data_loader.get_Fneu()
        stat = data_loader.get_stat()
        is_cell = data_loader.get_is_cell()
        # ops = data_loader.get_ops()
        spks = data_loader.get_spks()

        ## Preprocessing steps:
        # Remove overlapping cells across planes
        if num_planes > 1:
            overlapping_cells = correct_overlapping_cells_across_planes(stat, is_cell, num_planes)

        # Get F_cell and Fneu_cell only activity
        F_cell, Fneu_cell, spks_cell = get_cell_only_activity(F, Fneu, spks, is_cell, num_planes)

        # Get neuropil corrected F with neuropil coefficient
        assert len(F_cell) == len(
            Fneu_cell
        ), "Fcell and Fneu cell must be the same length"
        Fcorr = get_corrected_F(F_cell, Fneu_cell, num_planes, neucoeff)
        for ip in range(num_planes):
            assert len(F_cell[ip]) == len(Fcorr[ip])

        F_to_save = os.path.join(data_dir, "files", "F.npy")
        np.save(F_to_save, Fcorr)
        # S_to_save = os.path.join(data_dir, "files", "spks.npy")
        # np.save(S_to_save, spks_cell)

    # # # Plot multiple traces
    # # fig, axs = plt.subplots(8,1)
    # # for i in list(range(8)):
    # #     axs[i].plot(Fcorr[0][i])
    # # fig.savefig(os.path.join(result_dir, "example traces.eps"), format="eps")
    
    ## ----------------------------------------------------------------------------  
    # Load behavioral data and timestamps for images
    im_ts, last_imts = data_loader.get_im_ts()  # image time stamps in second
    
    if os.path.exists(os.path.join(file_dir, "event_df.pkl")):
        event_df = pd.read_pickle(os.path.join(file_dir, "event_df.pkl"))
    else:
        # # Correct `event_df` and imaging timestamps based on voltage recordings for Bruker
        if imaging_system == "Bruker":
            voltages = data_loader.get_voltages()  # Computer    
            event_df = data_loader.get_event_df()  # Arduino
            # # If file didn't save, replace the above with these two lines
            # event_df = extract_cues_from_voltages(voltages)
            # sio.savemat(os.path.join(file_dir, f"{animal}_cues.mat"), {'event_df': event_df})
            event_df, im_ts = correct_timestamps(event_df, im_ts, num_planes, imaging_system, voltages)   
        elif imaging_system == "INSS":
            # Correct event_df based on imaging timestamps for INSS
            event_df = data_loader.get_event_df()  # Arduino
            event_df = correct_timestamps(event_df, last_imts, num_planes, imaging_system)
        event_df.to_pickle(os.path.join(file_dir, "event_df.pkl"))

    # # Extract all event time points from new event_df
    [licks, CS1, CS2, CS3, sucrose, umami] = extract_events(event_df)
    allCS = filter_trials_by_minITI([CS1, CS2, CS3], post_cue_window)

    ## ----------------------------------------------------------------------------
    # Downsample Fcorr to 5hz if not already
    if os.path.exists(os.path.join(file_dir, "F_5hz.npy")) and os.path.exists(os.path.join(file_dir, "timestamps_5hz.npy")):
        Fcorr_5hz = np.load(os.path.join(file_dir, "F_5hz.npy"), allow_pickle=True)
        new_im_ts = np.load(os.path.join(file_dir,  "timestamps_5hz.npy"), allow_pickle=True)
    else:
        current_framerate = np.round(1 / ((im_ts[0][-1] - im_ts[0][0]) / len(im_ts[0]))).astype(
            int)
        if current_framerate != framerate:
            Fcorr_5hz, new_im_ts = downsample_data(Fcorr, im_ts, current_framerate, framerate)
            F_to_save = os.path.join(data_dir, "files", "F_5hz.npy")
            np.save(F_to_save, Fcorr_5hz)
            ts_to_save = os.path.join(data_dir, "files", "timestamps_5hz.npy") 
            np.save(ts_to_save, new_im_ts)    
        else:
            Fcorr_5hz = Fcorr
            new_im_ts = im_ts
            F_to_save = os.path.join(data_dir, "files", "F_5hz.npy")
            np.save(F_to_save, Fcorr_5hz)
            ts_to_save = os.path.join(data_dir, "files", "timestamps_5hz.npy") 
            np.save(ts_to_save, new_im_ts)    
    

    ## ----------------------------------------------------------------------------
    # # Normalize signal
    Fcorr_norm = normalize_signal(
        Fcorr_5hz, num_planes, "median"
    )  # can be z_score, median, robust_z_score

    # # # Extract Faround each cue in all cuetypes for each cell, shape is nCS_types x ntrials x nCell x nFrames
    F_around_cue = extract_F_around_events(
        allCS,
        Fcorr_norm,
        new_im_ts,
        num_planes,
        pre_cue_window,
        post_cue_window,
        binsize=None,
        framerate=framerate
    )
    file_to_save = os.path.join(data_dir, "files", "F_around_cue_raw.npy") 
    np.save(file_to_save, F_around_cue)  


    ## ----------------------------------------------------------------------------
    ## Plotting
    # behavior raster
    fig_rawplot = plot_raw_licks(allCS, licks, pre_cue_window, 10)
    plt.close(fig_rawplot)
    fig_rawplot.savefig(
        os.path.join(result_dir, "behavior_raster.png"), format="png"
    )
    

    # For plotting average PSTH around cue
    Fave_around_cue = extract_Fave_around_events(
        allCS,
        Fcorr_norm,
        new_im_ts,
        num_planes,
        pre_cue_window,
        post_cue_window,
    )
    # reshaping the data, output is nCell x nCS*nFrames
    n_cs, n_cells, n_time = Fave_around_cue.shape
    Fave_around_cue = Fave_around_cue.transpose(1, 0, 2).reshape(n_cells, n_cs * n_time)

    # Initialize parameters
    window_size = int(
        Fave_around_cue.shape[1] / len(trial_types)
    ) 
    framerate = np.round(1 / ((new_im_ts[0][-1] - new_im_ts[0][0]) / len(new_im_ts[0]))).astype(
            int)
    frames_to_reward = delay_to_reward * framerate
    pre_window_size = pre_cue_window * framerate
    sortwindow = [
        pre_window_size,
        pre_window_size + frames_to_reward,
    ]
    
    # # Plot PSTH
    fig_calcium_PSTH = plot_average_PSTH_around_interest_window(
        trial_types,
        Fave_around_cue,
        window_size,
        pre_window_size,
        frames_to_reward,
        sortwindow,
        framerate,
    )
    fig_calcium_PSTH.savefig(os.path.join(result_dir, "PSTH.png"), format="png")
    plt.close(fig_calcium_PSTH)

    # # Get the example cells based on sorted response, plot PSTH
    # example_cells = []
    # for cue_type in range(len(trial_types)):
    #     idx_sortresponse = np.argsort(
    #         np.mean(Fcorr_around_cue_down[:, cue_type*window_size+sortwindow[0]: cue_type*window_size+sortwindow[1]], axis=1)
    #     )[::-1]
    #     example_cells.extend(list(
    #         idx_sortresponse[: int(np.floor(0.01 * len(idx_sortresponse)))]
    #     ))
        
    # # # Plot individual cell activities
    # # plot_after_cue = 10  # only plot up to 6s after the cue
    # plot_individual_trial_average_activity(
    #     Fcorr_around_cue_down,
    #     trial_types,
    #     window_size,
    #     pre_window_size,
    #     frames_to_reward,
    #     example_cells,
    #     framerate,
    #     result_dir,
    # )
    
    # plot_individual_cells_activity(
    #     Fcorr_norm, allCS, im_idx_around_cue, example_cells, num_planes, plot_till_idx)
    # F_example_cells = F_around_cue[example_cells, :]
    # fig_calcium_PSTH_example_cells = plot_average_PSTH_around_interest_window(
    #     trial_types,
    #     F_example_cells,
    #     window_size,
    #     pre_window_size,
    #     frames_to_reward,
    #     sortwindow,
    #     framerate,
    # )
    # fig_calcium_PSTH_example_cells.savefig(
    #     os.path.join(result_dir, "PSTH_example_cells.png"), format="png"
    # )
    # plt.close(fig_calcium_PSTH_example_cells)



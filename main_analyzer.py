"""
READ BEFORE START:

    DATA format:
    - In each animal's file, create a subfoler for each day (called d1, d2 etc)
    - In each day's subfolder, have a folder called "files" containing all the recorded time points, voltage recording and behavior
    - suite2p folder is created automatically with suite2p, needs to have mat file for all information
    - can have the reference folder in this subfolder as well
    - running this file will make a result folder in this folder

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
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn.manifold import TSNE
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import (ModelDesc, EvalEnvironment, Term, EvalFactor, LookupFactor, dmatrices, INTERCEPT)
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import sys

from s2p_utils.data_loader import DataLoader
from s2p_utils.processing_utils import (
    correct_overlapping_cells_across_planes,
    correct_arduino_timestamps,
    get_cell_only_activity,
    extract_events,
    get_corrected_F,
    extract_interest_time_intervals,
    extract_imaging_ts_around_events,
    normalize_signal,
    extract_Fave_around_events,
)
from plot_utils import (
    plot_raw_licks,
    plot_average_PSTH_around_interest_window,
    plot_individual_cells_activity,
    plot_PC_screenplot,
    plot_PCs,
    make_silhouette_plot,
    plot_activity_clusters
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


def main():
    # Load data
    args = parse_args()
    data_loader = DataLoader(args.data_dir, args.num_planes)
    # Make a result folder if if didn't exist
    result_dir = os.path.join(args.data_dir, args.result_folder)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    file_dir = os.path.join(args.data_dir, "files")

    if os.path.exists(os.path.join(file_dir, "F.npy")):
        Fcorr_around_cue = np.load(os.path.join(file_dir, "F.npy"))
    else:
        # Load all necessary variables.
        F = data_loader.get_F()
        Fneu = data_loader.get_Fneu()
        stat = data_loader.get_stat()
        is_cell = data_loader.get_is_cell()
        # ops = data_loader.get_ops()
        # spks = data_loader.get_spks()
        event_df = data_loader.get_event_df()  # Arduino
        voltages = data_loader.get_voltages()  # Computer
        im_ts = data_loader.get_im_ts()  # image time stamps in second

        ## Preprocessing steps:
        # Remove overlapping cells across planes
        if args.num_planes > 1:
            overlapping_cells = correct_overlapping_cells_across_planes(stat, is_cell)

        # Get F_cell and Fneu_cell only activity
        F_cell, Fneu_cell = get_cell_only_activity(
            F, Fneu, is_cell, args.num_planes, args.cell_threshold
        )

        # Get neuropil corrected F with neuropil coefficient
        assert len(F_cell) == len(
            Fneu_cell
        ), "Fcell and Fneu cell must be the same length"
        Fcorr = get_corrected_F(F_cell, Fneu_cell, args.num_planes, args.neucoeff)
        for ip in range(args.num_planes):
            assert len(F_cell[ip]) == len(Fcorr[ip])

        # Normalization signal
        Fcorr_around_cue = normalize_signal(
            Fcorr, args.num_planes, "median"
        )  # can be z_score, median, robust_z_score

        # Correct `event_tf` timestamps.
        correct_arduino_timestamps(event_df, voltages)

        # Extract all event time points from new event_df
        [licks, CS1, CS2, CS3, sucrose, milk] = extract_events(event_df)
        allCS = [CS1, CS2, CS3]
        ## Plot behavior rasters
        fig_rawplot = plot_raw_licks(allCS, licks, args.pre_cue_window, 10)
        plt.close(fig_rawplot)
        fig_rawplot.savefig(
            os.path.join(result_dir, "behavior_raster.png"), format="png"
        )

        # Extract time around each cue and sorted by CS type
        interest_intervals = extract_interest_time_intervals(allCS, args)
        # Extract image time points around each cue and sorted by CS type and plane
        im_idx_around_cue = extract_imaging_ts_around_events(
            allCS, im_ts, args.num_planes, interest_intervals
        )

        # Extract average Fcorr around each cue in all cuetypes for each cell
        Fcorr_around_cue = extract_Fave_around_events(
            allCS,
            Fcorr,
            im_idx_around_cue,
            args.num_planes,
            args.pre_cue_window,
            args.post_cue_window,
        )
        Fcorr_around_cue = Fcorr_around_cue.transpose(1, 2, 0).reshape(
            Fcorr_around_cue.shape[1], -1, order="F"
        )  # reshaping the data to
        file_to_save = os.path.join(args.data_dir, "files", "F.npy")
        np.save(file_to_save, Fcorr_around_cue)

    # Initialize parameters
    window_size = int(Fcorr_around_cue.shape[1] / len(args.trial_types))
    framerate = args.framerate
    frames_to_reward = args.delay_to_reward * framerate
    pre_window_size = args.pre_cue_window * framerate
    sortwindow = [
        pre_window_size,
        pre_window_size + frames_to_reward,
    ]

    # Plot PSTH
    # fig_calcium_PSTH = plot_average_PSTH_around_interest_window(
    #     args.trial_types,
    #     Fcorr_around_cue,
    #     window_size,
    #     pre_window_size,
    #     frames_to_reward,
    #     sortwindow,
    #     framerate,
    # )
    # fig_calcium_PSTH.savefig(os.path.join(result_dir, "PSTH.png"), format="png")
    # plt.close(fig_calcium_PSTH)

    # # Get the example cells based on sorted response, plot PSTH
    # sortresponse = np.argsort(np.mean(Fcorr_around_cue[:, sortwindow[0] : sortwindow[1]], axis=1))[
    #     ::-1
    # ]
    # example_cells = list(sortresponse[:200]) + list(sortresponse[-200:])
    # F_example_cells = Fcorr_around_cue[example_cells, :]
    # fig_calcium_PSTH_example_cells = plot_average_PSTH_around_interest_window(
    #     args.trial_types,
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

    # PCA
    populationdata = Fcorr_around_cue
    pca = PCA(n_components=populationdata.shape[1], whiten=True)
    pca.fit(populationdata)
    transformed_data = pca.transform(populationdata)
    np.save(os.path.join(file_dir,'transformed_data.npy'), transformed_data) 

    pca_vectors = pca.components_
    print("Number of PCs = %d" % (pca_vectors.shape[0]))
    x = 100 * pca.explained_variance_ratio_
    xprime = x - (x[0] + (x[-1] - x[0]) / (x.size - 1) * np.arange(x.size))
    num_retained_pcs = np.argmin(xprime)
    print("Number of PCs to keep = %d" % (num_retained_pcs))
    
    # fig_pc_screenplot = plot_PC_screenplot(pca, num_retained_pcs)
    # fig_pc_screenplot.savefig(os.path.join(result_dir, "PC_screenplot.png"), format="png")

    # fig_pcs = plot_PCs(
    #     pca_vectors,
    #     num_retained_pcs,
    #     args.trial_types,
    #     window_size,
    #     pre_window_size,
    #     frames_to_reward,
    #     framerate,
    # )
    # fig_pcs.savefig(os.path.join(result_dir, "PCs.png"), format="png")


    ## Clustering 
    max_n_clusters = 11
    possible_n_clusters = np.arange(2, max_n_clusters+1)
    possible_n_nearest_neighbors = np.array([100, 500, 1000])    
    silhouette_scores = np.nan*np.ones((possible_n_clusters.size,
                                        possible_n_nearest_neighbors.size))

    for n_clustersidx, n_clusters in enumerate(possible_n_clusters):
        for nnidx, nn in enumerate(possible_n_nearest_neighbors):
            model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=nn)
            model.fit(transformed_data[:,:num_retained_pcs])
            silhouette_scores[n_clustersidx, nnidx] = silhouette_score(transformed_data[:,:num_retained_pcs],
                                                                    model.labels_,
                                                                    metric='cosine')
            print( 'Done with numclusters = %d, num nearest neighbors = %d: score = %.3f'%(n_clusters,
                                                                                        nn,
                                                                                        silhouette_scores[n_clustersidx,                                                                           
                                                                                                            nnidx]))
    print('Done with model fitting')

    temp = {}
    temp['possible_n_clusters'] = possible_n_clusters
    temp['possible_n_nearest_neighbors'] = possible_n_nearest_neighbors
    temp['silhouette_scores'] = silhouette_scores
    temp['shape'] = 'cluster_nn'
    with open(os.path.join(file_dir, 'silhouette_scores.pickle'), 'wb') as f:
        pickle.dump(temp, f)    
    with open(os.path.join(file_dir, 'silhouette_scores.pickle'), 'rb') as f:
        silhouette_scores = pickle.load(f)
        
    # transformed_data = np.load(os.path.join(file_dir, 'transformed_data.npy'))

    # Identify optimal parameters from the above parameter space
    temp = np.where(silhouette_scores['silhouette_scores']==np.nanmax(silhouette_scores['silhouette_scores']))
    n_clusters = silhouette_scores['possible_n_clusters'][temp[0][0]]
    n_nearest_neighbors = silhouette_scores['possible_n_nearest_neighbors'][temp[1][0]]

    print(n_clusters, n_nearest_neighbors)

    # Redo clustering with these optimal parameters
    model = SpectralClustering(n_clusters=n_clusters,
                            affinity='nearest_neighbors',
                            n_neighbors=n_nearest_neighbors)

    # model = KMeans(n_clusters=n_clusters)

    # model = AgglomerativeClustering(n_clusters=9,
    #                                 affinity='l1',
    #                                 linkage='average')

    model.fit(transformed_data[:,:num_retained_pcs])

    temp = silhouette_score(transformed_data[:,:num_retained_pcs], model.labels_, metric='cosine')

    print('Number of clusters = %d, average silhouette = %.3f'%(len(set(model.labels_)), temp))

    # Save this optimal clustering model.
    # with open(os.path.join(basedir, 'clusteringmodel.pickle'), 'wb') as f:
    #     pickle.dump(model, f)

            
    # Since the clustering labels are arbitrary, I rename the clusters so that the first cluster will have the most
    # positive response and the last cluster will have the most negative response.
    def reorder_clusters(rawlabels):
        uniquelabels = list(set(rawlabels))
        responses = np.nan*np.ones((len(uniquelabels),))
        for l, label in enumerate(uniquelabels):
            responses[l] = np.mean(populationdata[rawlabels==label, pre_window_size:2*pre_window_size])
        temp = np.argsort(responses).astype(int)[::-1]
        temp = np.array([np.where(temp==a)[0][0] for a in uniquelabels])
        outputlabels = np.array([temp[a] for a in list(np.digitize(rawlabels, uniquelabels)-1)])
        return outputlabels
    
    newlabels = reorder_clusters(model.labels_)

    # Create a new variable containing all unique cluster labels
    uniquelabels = list(set(newlabels))

    # np.save(os.path.join(basedir, 'OFCCaMKII_clusterlabels.npy'), newlabels)
    
    # fig_silouette = make_silhouette_plot(transformed_data[:,:num_retained_pcs], model.labels_)
    # fig_silouette.savefig(os.path.join(result_dir, 'silouette.png'), format='png')
    cmax = 0.1
    sortwindow = [15, 100]
    # fig_activity_cluster = plot_activity_clusters(populationdata, uniquelabels, newlabels, args.trial_types, sortwindow, window_size, pre_window_size, frames_to_reward, framerate)
    # fig_activity_cluster.savefig(os.path.join(result_dir, 'activity_clusters.png'), format='png')

    num_clusterpairs = len(uniquelabels)*(len(uniquelabels)-1)/2
    colors_for_cluster = [[0.933, 0.250, 0.211],
                        [0.941, 0.352, 0.156],
                        [0.964, 0.572, 0.117],
                        [0.980, 0.686, 0.250],
                        [0.545, 0.772, 0.247],
                        [0.215, 0.701, 0.290],
                        [0, 0.576, 0.270],
                        [0, 0.650, 0.611],
                        [0.145, 0.662, 0.878],
                        [0.604, 0.055, 0.918]]
    numrows = int(np.ceil(num_clusterpairs**0.5))
    numcols = int(np.ceil(num_clusterpairs/np.ceil(num_clusterpairs**0.5)))
    fig, axs = plt.subplots(numrows, numcols, figsize=(3*numrows, 3*numcols))

    tempsum = 0
    for c1, cluster1 in enumerate(uniquelabels):
        for c2, cluster2 in enumerate(uniquelabels):
            if cluster1>=cluster2:
                continue
            temp1 = transformed_data[np.where(newlabels==cluster1)[0], :num_retained_pcs]
            temp2 = transformed_data[np.where(newlabels==cluster2)[0], :num_retained_pcs]
            X = np.concatenate((temp1, temp2), axis=0)
            tsne = TSNE(n_components=2, init='random',
                        random_state=0, perplexity=100)
            Y = tsne.fit_transform(X)
            ax = axs[int(np.round(tempsum/numcols)),
                     int(np.round(tempsum - tempsum/numcols*numcols))]   
            ax.scatter(Y[:np.sum(newlabels==cluster1),0],
                    Y[:np.sum(newlabels==cluster1),1],
                    color=colors_for_cluster[cluster1], label='Cluster %d'%(cluster1+1), alpha=1)
            ax.scatter(Y[np.sum(newlabels==cluster1):,0],
                    Y[np.sum(newlabels==cluster1):,1],
                    color=colors_for_cluster[cluster2], label='Cluster %d'%(cluster2+1), alpha=1)

            ax.set_xlabel('tsne dimension 1')
            ax.set_ylabel('tsne dimension 2')
            ax.legend()
            tempsum += 1
    fig.tight_layout()        
        
        
        

if __name__ == "__main__":
     main()

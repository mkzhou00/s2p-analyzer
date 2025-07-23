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

def build_knn(X, k):
    """Return sparse k-NN connectivity graph (CSR)."""
    return kneighbors_graph(
        X, k,
        mode="connectivity",
        include_self=True,    # matches SpectralClustering default
        n_jobs=-1
    )

logger = logging.getLogger(__name__)


# Initialize parameters
data_dir = "Z:\\2p\\experiment1\\"
learning_stage = "late"
pre_cue_window = 3
post_cue_window = 17
delay_to_reward = 3
min_cell_prob = 0.5
neucoeff = 0.7
framerate = 5
target_frames = 300 # Should be the total time window * framerate * nCS, so 300 for the current settings
trial_types = ["CS1+", "CS2+", "CS3-"]
learning_stage = 'early'
cluster_algorithm = "Discretization" #KMeans

# Set animals and days for early and late learning
if learning_stage == "early":
    result_dir = "Z:\\2p\\experiment1\\population_data\\early learning\\"
    animal_list = [
        "MZ_CA1_WD_F3\\d1",
        "MZ_CA1_WD_M4\\d1",
        "MZ_CA1_WD_M5\\d1",
        "MZ_CA1_WD_M6\\d2",
        "MZ_CA1_WD_M7\\d1",
        "MZ_CA1_WD_M8\\d1",
        "MZ_CA1_WD_JB_54\\d3",
        "MZ_CA1_WD_JB_55\\d3"
    ]
elif learning_stage == "late":
    result_dir = "Z:\\2p\\experiment1\\population_data\\late learning\\"
    animal_list = [
        "MZ_CA1_WD_F3\\d7",
        "MZ_CA1_WD_M4\\d5",
        "MZ_CA1_WD_M5\\d6",
        "MZ_CA1_WD_M6\\d6",
        "MZ_CA1_WD_M7\\d5",
        "MZ_CA1_WD_M8\\d6",
        "MZ_CA1_WD_JB_54\\d8",
        "MZ_CA1_WD_JB_55\\d12"
    ]   

# For plotting
window_size = 100
frames_to_reward = delay_to_reward * framerate
pre_window_size = pre_cue_window * framerate
        
# Load and concatenate population data across animals
populationdata, animal_id = load_population_data(animal_list, data_dir, result_dir, trial_types, target_frames, window_size, subtrials=10)

# Define cache file paths before running PCA
pca_path = os.path.join(result_dir, "pca_model.pickle")
clustering_path = os.path.join(result_dir, "clustering_model.pickle")
labels_path = os.path.join(result_dir, "clusterlabels.npy")
transformed_path = os.path.join(result_dir, "transformed_data.npy")

## -------------------------------------------------------------------------------------------------------
## STEP 1: Dimensionality reduction with PCA
# Load saved data and models if exist
if all(os.path.exists(p) for p in [pca_path, transformed_path]):
    with open(pca_path, "rb") as f:
        pca_data = pickle.load(f)
        pca = pca_data["pca"]
        num_retained_pcs = pca_data["num_retained_pcs"]
        
    transformed_data = np.load(transformed_path)   
else:
    pca = PCA(n_components=min(populationdata.shape[0], populationdata.shape[1]), whiten=True)
    pca.fit(populationdata)
    pca_vectors = pca.components_
    x = 100 * pca.explained_variance_ratio_
    xprime = x - (x[0] + (x[-1] - x[0]) / (x.size - 1) * np.arange(x.size))
    num_retained_pcs = np.argmin(xprime)

    # # dimension-reduced data on the first principal components
    transformed_data = pca.transform(populationdata)
    np.save(os.path.join(result_dir, "transformed_data.npy"), transformed_data)
    
    # Save PCA model
    with open(pca_path, "wb") as f:
        pickle.dump({
            "pca": pca,
            "num_retained_pcs": num_retained_pcs
        }, f)       
    
    # Plot PC screen plot
    fig_pc_screenplot = plot_PC_screenplot(pca, x, num_retained_pcs)
    fig_pc_screenplot.savefig(
        os.path.join(result_dir, "PC_screenplot.png"), format="png"
    )
    plt.close(fig_pc_screenplot)

    # Plot PCs
    fig_pcs = plot_PCs(
        pca_vectors,
        num_retained_pcs,
        trial_types,
        window_size,
        pre_window_size,
        frames_to_reward,
        framerate,
    )
    fig_pcs.savefig(os.path.join(result_dir, "PCs.png"), format="png")
    plt.close(fig_pcs)

## -------------------------------------------------------------------------------------------------------
## STEP 2: Clustering
if all(os.path.exists(p) for p in [clustering_path, labels_path]):    
    
    with open(clustering_path, "rb") as f:
        cluster_data = pickle.load(f)
        model = cluster_data["model"]
        n_clusters = cluster_data["n_clusters"]
        n_nearest_neighbors = cluster_data["n_neighbors"]
    
    newlabels = np.load(labels_path)
    uniquelabels = list(set(newlabels))
    
else:    
    max_n_clusters = 9  # initialize nclusters, can run more but takes longer, 9 is relatively optimal
    # iTers = 100
    possible_n_clusters = np.arange(2, max_n_clusters + 1)  # has to be at least two
    possible_n_nearest_neighbors = np.array(
        [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500]
    )  # depends on the size of the data, for a data with 300 neurons, the above is fine
    
    silhouette_scores = np.nan*np.ones((possible_n_clusters.size,
                                        possible_n_nearest_neighbors.size,
                                        ))
    discrete_labels = np.nan*np.ones((possible_n_clusters.size,
                            possible_n_nearest_neighbors.size,
                            transformed_data.shape[0]))
    
    
    # Fit clusters with Spectral Clustering
    for nnidx, nn in enumerate(possible_n_nearest_neighbors):
        G = build_knn(transformed_data[:,:num_retained_pcs], nn)

        for n_clustersidx, n_clusters in enumerate(possible_n_clusters):
            # model = SpectralClustering(
            #     n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=nn
            # )  # separate clusters based on n-nearest neighbors
            # model.fit(transformed_data[:, :num_retained_pcs])
            # silhouette_scores[n_clustersidx, nnidx] = silhouette_score(
            #     transformed_data[:, :num_retained_pcs], model.labels_, metric="cosine"
            # )  
            # silhouette coeff is  calculated as (mean near-cluster distance - mean intra-cluster distance) / max of the two, 
            # 1 is the best, -1 is the worst
            embed = spectral_embedding(
                    G,
                    n_components=n_clusters,
                    eigen_solver="arpack",      # use same solver as SC
                    drop_first=False            # keep all dimensions
                )

            # for N in range(iTers):
            labels = discretize(embed, random_state=None)
            discrete_labels[n_clustersidx, nnidx,:] = labels
            silhouette_scores[n_clustersidx, nnidx] = silhouette_score(transformed_data[:,:num_retained_pcs],
                                                                    labels,
                                                                    metric='cosine') 

            print('Done with numclusters = %d, num nearest neighbors = %d: score = %.3f'%(n_clusters,
                                                                                        nn,
                                                                                        np.nanmean(silhouette_scores[
                                                                                        n_clustersidx,
                                                                                        nnidx])))
    print("Done with model fitting")

    temp = {
        "possible_n_clusters": possible_n_clusters,
        "possible_n_nearest_neighbors": possible_n_nearest_neighbors,
        "silhouette_scores": silhouette_scores,
        "shape": "cluster_nn"
    }
    
    # Save the silhouette scores
    with open(os.path.join(result_dir, "silhouette_scores.pickle"), "wb") as f:
        pickle.dump(temp, f)
    # with open(os.path.join(result_dir, "silhouette_scores.pickle"), "rb") as f:
    #     silhouette_scores = pickle.load(f)
        
    # Identify optimal parameters from the above parameter space
    # temp = np.where(
    #     silhouette_scores["silhouette_scores"]== np.nanmax(silhouette_scores"silhouette_scores"]))
    # n_clusters = silhouette_scores["possible_n_clusters"][temp[0][0]]
    # n_nearest_neighbors = silhouette_scores["possible_n_nearest_neighbors"][temp[1][0]]
    
    
    temp = np.where(silhouette_scores==np.nanmax(silhouette_scores))
    n_clusters = temp[0][0]+2 
    n_nearest_neighbors = possible_n_nearest_neighbors[temp[1][0]]    
    
    print("Optimal number of clusters:",
          n_clusters,
          "; Optimal neighbors:",
          n_nearest_neighbors)
    
## -------------------------------------------------------------------------------------------------------
    ## STEP 3: Redo clustering with these optimal parameters
    # model = SpectralClustering(
    #     n_clusters=n_clusters,
    #     affinity="nearest_neighbors",
    #     n_neighbors=n_nearest_neighbors,
    # )
    # model = KMeans(n_clusters=n_clusters)
    # model = AgglomerativeClustering(n_clusters=9,
    #                                 affinity='l1',
    #                                 linkage='average')
    model = SpectralClustering(n_clusters=n_clusters, 
                           affinity='nearest_neighbors', 
                           n_neighbors=n_nearest_neighbors,
                           assign_labels='discretize',
                           eigen_solver='arpack',
                           n_jobs=-1)
    model.fit(transformed_data[:, :num_retained_pcs])

    temp = silhouette_score(
        transformed_data[:, :num_retained_pcs], model.labels_, metric="cosine"
    )
    print(
        "Number of clusters = %d, average silhouette = %.3f"
        % (len(set(model.labels_)), temp)
    )

    # # Save this optimal clustering model.
    with open(clustering_path, "wb") as f:
        pickle.dump({
            "model": model,
            "n_clusters": n_clusters,
            "n_neighbors": n_nearest_neighbors
        }, f)

    # Rename the clusters so that the first cluster will have the most
    # positive response and the last cluster will have the most negative response.
    newlabels = reorder_clusters(populationdata, pre_window_size, model.labels_)
    # Create a new variable containing all unique cluster labels
    uniquelabels = list(set(newlabels))
    np.save(os.path.join(result_dir, "clusterlabels.npy"), newlabels)

## -------------------------------------------------------------------------------------------------------
# Plots
# Plot animal's contribution for each cluster
df_contrib_by_animal = pd.DataFrame(
    {'Animal': animal_id,
        'Cluster': newlabels}
)
cluster_counts = df_contrib_by_animal.groupby(['Cluster', 'Animal']).size().unstack(fill_value=0)
cluster_percent = cluster_counts.divide(cluster_counts.sum(axis=1), axis=0)
print("\nNeuron counts per cluster per animal:")
print(cluster_counts)
# Save as CSV
cluster_counts.to_csv(os.path.join(result_dir, "cluster_counts.csv"))
cluster_percent.to_csv(os.path.join(result_dir, "cluster_percent.csv"))

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_percent, annot=True, cmap="viridis", fmt=".2f")
plt.title("Proportion of neurons from each animal in each cluster")
plt.ylabel("Cluster")
plt.xlabel("Animal")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "cluster_contribution_heatmap.png"))
plt.show()


# Plot silhouette coefficient scores for each cluster
fig_silouette = make_silhouette_plot(
    transformed_data[:, :num_retained_pcs], model.labels_
)
fig_silouette.savefig(os.path.join(result_dir, "silouette.png"), format="png")
plt.close(fig_silouette)


# Plot activity clusters under each CS
fig_activity_cluster = plot_activity_clusters(
    populationdata,
    uniquelabels,
    newlabels,
    trial_types,
    [15, 100],
    window_size,
    pre_window_size,
    frames_to_reward,
    framerate,
)
fig_activity_cluster.savefig(
    os.path.join(result_dir, "activity_clusters.png"), format="png"
)
plt.close(fig_activity_cluster)

# Plot all cluster pairs
fig_cluster_pairs = plot_cluster_pairs(
    transformed_data, uniquelabels, newlabels, num_retained_pcs
)
fig_cluster_pairs.savefig(os.path.join(result_dir, "clusters.png"), format="png")
plt.close(fig_cluster_pairs)



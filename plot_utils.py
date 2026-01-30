import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os
from sklearn.metrics import (
    accuracy_score,
    silhouette_score,
    adjusted_rand_score,
    silhouette_samples,
)
from sklearn.manifold import TSNE
import math
from common_func import (
    central_tendency,
    standardize_plot_graphics,
    CDFplot,
    t_test,
    convert_pvalue_to_asterisks,
    point_to_line,
    getCumsumChangePoint,
    plot_raw_licks,
    getCumsumChangePoint,
)

sns.set_style("ticks")
import matplotlib as mpl
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import to_rgba

mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.labelspacing"] = 0.2
mpl.rcParams["axes.labelpad"] = 2
mpl.rcParams["xtick.major.size"] = 2
mpl.rcParams["xtick.major.width"] = 0.5
mpl.rcParams["xtick.major.pad"] = 1
mpl.rcParams["ytick.major.size"] = 2
mpl.rcParams["ytick.major.width"] = 0.5
mpl.rcParams["ytick.major.pad"] = 1
mpl.rcParams["lines.scale_dashes"] = False
mpl.rcParams["lines.dashed_pattern"] = (2, 1)
mpl.rcParams["font.sans-serif"] = ["Helvetica LT Std"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.color"] = "k"


def tsplot(data, ax, color, label, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd / np.sqrt(data.shape[0]), est + sd / np.sqrt(data.shape[0]))
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=color, **kw)
    ax.plot(x, est, color=color, label=label, **kw)
    ax.margins(x=0)
    return ax


def standardize_plot_graphics(ax):
    [i.set_linewidth(0.5) for i in ax.spines.values()]
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def plot_raw_licks(CS, licks, before, after):
    fig_rawplot, ax = plt.subplots(1, len(CS), figsize=(3 * (len(CS)), 3))
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for cs_type, (cs, a) in enumerate(zip(CS, ax)):
        if cs_type == 2:
            cs_sign = "-"
        else:
            cs_sign = "+"
        for i in range(0, len(cs)):
            raw_licks = licks[
                (licks >= ((cs[i]) - before)) & (licks <= (cs[i] + after))
            ]
            a.vlines(
                x=(raw_licks - cs[i]),
                ymin=i,
                ymax=(i + 1),
                linewidth=1,
                color="#654321",
            )
            if i == len(cs) - 1:
                a.spines["top"].set_visible(False)
                a.spines["right"].set_visible(False)
                a.set_xlim(-before, after)
                a.set_ylim(0, len(cs))
                a.vlines(
                    x=0,
                    ymin=0,
                    ymax=len(cs),
                    linestyles="dashed",
                    color="k",
                    linewidth=1,
                )
                a.vlines(
                    x=3,
                    ymin=0,
                    ymax=len(cs),
                    linestyles="dashed",
                    color="k",
                    linewidth=1,
                )
                a.set_xlabel("Time (s)")
                a.set_title(("CS" + str(cs_type + 1) + cs_sign))
        a.set_ylim((0, len(cs)))
    ax[0].set_ylabel("Trials")
    return fig_rawplot


def plot_average_PSTH_around_interest_window(
    CStrials: list,
    F,
    window_size,
    pre_window_size,
    frames_to_reward,
    sortwindow,
    framerate,
):
    fig_PSTH, axs = plt.subplots(
        1,
        len(CStrials),
        figsize=(3 * len(CStrials), 4),
        sharex="all",
        sharey="row",
    )
    cbar_ax = fig_PSTH.add_axes([0.91, 0.3, 0.01, 0.4])
    cbar_ax.tick_params(width=0.5)

    sortresponse = np.argsort(np.mean(F[:, sortwindow[0] : sortwindow[1]], axis=1))[
        ::-1
    ]
    cmin = np.amin(F[0])
    cmax = np.amax(F[0])

    for cue_type in range(len(CStrials)):
        axs[cue_type].set_title(CStrials[cue_type])
        ax = axs[cue_type]
        sns.heatmap(
            F[sortresponse, cue_type * window_size : (cue_type + 1) * window_size],
            ax=ax,
            cmap=plt.get_cmap("coolwarm"),
            vmax=max(cmax, 0.1),
            vmin=min(-cmax, -0.1),
            cbar=(cue_type == 0),
            cbar_ax=cbar_ax if (cue_type == 0) else None,
            cbar_kws={"label": "Normalized fluorescence"},
        )
        ax.grid(False)
        ax.tick_params(width=0.5)
        ax.set_xticks(
            [0, pre_window_size, pre_window_size + frames_to_reward, window_size]
        )
        ax.set_xticklabels(
            [
                str(int((a - pre_window_size + 0.0) / framerate))
                for a in [
                    0,
                    pre_window_size,
                    pre_window_size + frames_to_reward,
                    window_size,
                ]
            ]
        )
        ax.set_yticks([])
        ax.axvline(pre_window_size, linestyle="--", color="k", linewidth=0.5)
        ax.axvline(
            pre_window_size + frames_to_reward, linestyle="--", color="k", linewidth=0.5
        )
        ax.set_xlabel("Time from cue (s)")
        axs[0].set_ylabel("Neurons")

        fig_PSTH.tight_layout()
        fig_PSTH.subplots_adjust(right=0.90)

    return fig_PSTH


def plot_individual_cells_activity(
    F, CS, im_idx_around_cue, cells_to_plot, num_planes: int, plot_till_idx=-1
):
    cells_per_plane = [[] for _ in range(num_planes)]
    for cell in cells_to_plot:
        if cell < len(F[0]):
            cells_per_plane[0].append(cell)
        elif (cell < (len(F[0]) + len(F[1]))) and (cell >= len(F[0])):
            cell = cell - len(F[0])
            cells_per_plane[1].append(cell)
        # correct for cell number up to 4 planes
        # elif (cell < (len(F[0]) + len(F[1]) + len(F[2]))) and (cell >= (len(F[0]) + len(F[1]))):
        #     cell = cell - (len(F[0]) + len(F[1]))
        #     cells_per_plane[2].append(cell)
        # elif (cell < (len(F[0]) + len(F[1]) + len(F[2]) + len(F[3]))) and (cell >= (len(F[0]) + len(F[1]) + len(F[2]))):
        #     cell = cell - (len(F[0]) + len(F[1]) + len(F[2]))
        #     cells_per_plane[3].append(cell)

    for ip in range(num_planes):
        for cell in cells_per_plane[ip]:
            fig_cell, axs = plt.subplots(len(CS[0]), len(CS), figsize=(6, 6))
            for cue_type, cs in enumerate(CS):
                cue_ts = im_idx_around_cue[ip][cue_type]
                if cue_type == 0:
                    flattened_cue_ts = [item for sublist in cue_ts for item in sublist]
                    ymax = np.max(F[ip][cell][flattened_cue_ts])
                    ymin = np.min(F[ip][cell][flattened_cue_ts])
                for trial in range(len(CS[0])):
                    ax = axs[trial, cue_type]
                    Ftemp = F[ip][cell][cue_ts[trial][:plot_till_idx]]
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    if cue_type != 0:
                        ax.yaxis.set_major_locator(ticker.NullLocator())
                    ax.spines["bottom"].set_visible(False)
                    ax.axvline(30, linestyle="--", color="k", linewidth=0.5)
                    ax.axvline(40, linestyle="--", color="k", linewidth=0.5)
                    ax.axvline(60, linestyle="--", color="k", linewidth=0.5)
                    ax.plot(Ftemp, linewidth=0.5, linestyle="-", color="blue")
                    ax.set_ylim([ymin, ymax])
                    ax.xaxis.set_major_locator(ticker.NullLocator())
                    ax.set_yticks([round(ymin), round(ymax)])
                    ax.set_yticklabels([round(ymin), round(ymax)], fontsize=8)
                    if trial == (len(CS[0]) - 1):
                        ax.spines["bottom"].set_visible(True)
                        ax.set_xticks([0, 30, 60, 120])
                        ax.set_xticklabels(
                            [str(int((a - 30) / 10)) for a in [0, 30, 60, 120]]
                        )
                        ax.set_xlabel("Time from cue (s)")
            axs[0, 0].title.set_text("CS1+")
            axs[0, 1].title.set_text("CS2+")
            axs[0, 2].title.set_text("CS3-")
            # fig_cell.tight_layout()
            plt.show(fig_cell)


def plot_individual_trial_average_activity(
    F, 
    CStrials,     
    window_size,
    pre_window_size,
    frames_to_reward,
    cells_to_plot, 
    framerate,
    result_dir, 
    post_window_size=17
):
    # cells_per_plane = [[] for _ in range(num_planes)]
    # for cell in cells_to_plot:
    #     if cell < len(F[0]):
    #         cells_per_plane[0].append(cell)
    #     elif (cell < (len(F[0]) + len(F[1]))) and (cell >= len(F[0])):
    #         cell = cell - len(F[0])
    #         cells_per_plane[1].append(cell)
        # correct for cell number up to 4 planes
        # elif (cell < (len(F[0]) + len(F[1]) + len(F[2]))) and (cell >= (len(F[0]) + len(F[1]))):
        #     cell = cell - (len(F[0]) + len(F[1]))
        #     cells_per_plane[2].append(cell)
        # elif (cell < (len(F[0]) + len(F[1]) + len(F[2]) + len(F[3]))) and (cell >= (len(F[0]) + len(F[1]) + len(F[2]))):
        #     cell = cell - (len(F[0]) + len(F[1]) + len(F[2]))
        #     cells_per_plane[3].append(cell)
    actual_frames_to_plot = pre_window_size +  post_window_size * framerate
    for cell in cells_to_plot:
        fig_cell, axs = plt.subplots(1, len(CStrials), figsize=(6,3), dpi=100)
        for cue_type in range(len(CStrials)):
            ymax = np.max(F[cell])
            ymin = np.min(F[cell])
            ax = axs[cue_type]
            Ftemp= F[cell, cue_type*window_size : cue_type*window_size+actual_frames_to_plot]
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if cue_type != 0:
                ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.spines["bottom"].set_visible(False)
            ax.axvline(pre_window_size, linestyle="--", color="k", linewidth=0.5)
            ax.axvline(pre_window_size + int(frames_to_reward*1/3), linestyle="--", color="k", linewidth=0.5)
            ax.axvline(pre_window_size + frames_to_reward, linestyle="--", color="k", linewidth=0.5)
            ax.plot(Ftemp, linewidth=0.5, linestyle="-", color="blue")
            ax.set_ylim([ymin, ymax])
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.set_yticks([round(ymin, 2), round(ymax, 2)])
            ax.set_yticklabels([round(ymin, 2), round(ymax, 2)], fontsize=8)
            ax.spines["bottom"].set_visible(True)
            ax.set_xticks(
            [0, pre_window_size, pre_window_size + frames_to_reward, actual_frames_to_plot]
        )
            ax.set_xticklabels(
                [
                    str(int((a - pre_window_size + 0.0) / framerate))
                    for a in [
                        0,
                        pre_window_size,
                        pre_window_size + frames_to_reward,
                        actual_frames_to_plot,
                    ]
                ]
            )
        # ax.set_yticks([])
        ax.axvline(pre_window_size, linestyle="--", color="k", linewidth=0.5)
        ax.axvline(
            pre_window_size + frames_to_reward, linestyle="--", color="k", linewidth=0.5
        )
        axs[1].set_xlabel("Time from cue (s)")
        axs[0].set_ylabel("Normalized signal")
        axs[0].set_title("CS1+", fontsize=9)
        axs[1].set_title("CS2+", fontsize=9)
        axs[2].set_title("CS3-", fontsize=9)
        # fig_cell.tight_layout()
        fig_cell.savefig(os.path.join(result_dir, f"Cell_{cell}.png"), format="png")
        plt.close(fig_cell)


def plot_PC_screenplot(pca, x, num_retained_pcs):

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(np.arange(pca.explained_variance_ratio_.shape[0]).astype(int) + 1, x, "k")
    ax.set_ylabel("Percentage of\nvariance explained")
    ax.set_xlabel("PC number")
    ax.axvline(num_retained_pcs, linestyle="--", color="k", linewidth=0.5)
    ax.set_title("Scree plot")
    # ax.set_xlim([0,50])
    [i.set_linewidth(0.5) for i in ax.spines.values()]
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.subplots_adjust(left=0.3)
    fig.subplots_adjust(right=0.98)
    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(top=0.9)

    return fig


def plot_PCs(
    pca_vectors,
    num_retained_pcs,
    trial_types,
    window_size,
    pre_window_size,
    frames_to_reward,
    framerate,
):

    colors_for_key = {}
    colors_for_key["CS1+"] = (0, 0.5, 1)
    colors_for_key["CS2+"] = (1, 0.5, 0)
    colors_for_key["CS3-"] = (0.8, 0.8, 0.8)

    numcols = 3.0
    fig, axs = plt.subplots(
        int(np.ceil(num_retained_pcs / numcols)),
        int(numcols),
        sharey="all",
        figsize=(2 * numcols, 2 * int(np.ceil(num_retained_pcs / numcols))),
    )
    for pc in range(num_retained_pcs):
        ax = axs.flat[pc]
        for k, tempkey in enumerate(trial_types):
            ax.plot(
                pca_vectors[pc, k * window_size : (k + 1) * window_size],
                color=colors_for_key[tempkey],
                label="PC %d: %s" % (pc + 1, tempkey),
            )
        ax.axvline(pre_window_size, linestyle="--", color="k", linewidth=1)
        ax.annotate(
            s="PC %d" % (pc + 1),
            xy=(0.45, 0.06),
            xytext=(0.45, 0.06),
            xycoords="axes fraction",
            textcoords="axes fraction",
            multialignment="center",
            size="large",
        )
        if pc >= num_retained_pcs - numcols:
            ax.set_xticks(
                [0, pre_window_size, pre_window_size + frames_to_reward, window_size]
            )
            ax.set_xticklabels(
                [
                    str(int((a - pre_window_size + 0.0) / framerate))
                    for a in [
                        0,
                        pre_window_size,
                        pre_window_size + frames_to_reward,
                        window_size,
                    ]
                ]
            )
        else:
            ax.set_xticks([])
            ax.xaxis.set_ticks_position("none")
        if pc % numcols:
            ax.yaxis.set_ticks_position("none")
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    fig.text(
        0.5,
        0.05,
        "Time from cue (s)",
        horizontalalignment="center",
        rotation="horizontal",
    )
    fig.text(0.02, 0.6, "PCA weights", verticalalignment="center", rotation="vertical")
    fig.tight_layout()
    for ax in axs.flat[num_retained_pcs:]:
        ax.set_visible(False)

    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.13)
    return fig


def make_silhouette_plot(X, cluster_labels):
    colors_for_cluster = [
        [0.933, 0.250, 0.211],
        [0.941, 0.352, 0.156],
        [0.964, 0.572, 0.117],
        [0.980, 0.686, 0.250],
        [0.545, 0.772, 0.247],
        [0.215, 0.701, 0.290],
        [0, 0.576, 0.270],
        [0, 0.650, 0.611],
        [0.145, 0.662, 0.878],
        [0.604, 0.055, 0.918],
    ]

    n_clusters = len(set(cluster_labels))

    fig_silhouette, ax = plt.subplots(1, 1)
    fig_silhouette.set_size_inches(4, 4)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax.set_xlim([-0.4, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(X, cluster_labels, metric="cosine")

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric="cosine")

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors_for_cluster[i]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.9,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    return fig_silhouette


def plot_activity_clusters(
    populationdata,
    uniquelabels,
    newlabels,
    trial_types,
    sortwindow,
    window_size,
    pre_window_size,
    frames_to_reward,
    framerate,
):
    colors_for_key = {}
    colors_for_key["CS1+"] = (0, 0.5, 1)
    colors_for_key["CS2+"] = (1, 0.5, 0)
    colors_for_key["CS3-"] = (0.5, 0.5, 0.5)

    fig_activity_cluster, axs = plt.subplots(
        len(trial_types) + 1,
        len(uniquelabels),
        figsize=(2 * len(uniquelabels), 2 * (len(trial_types) + 1)),
    )
    cbar_ax = fig_activity_cluster.add_axes([0.94, 0.3, 0.01, 0.4])
    cbar_ax.tick_params(width=0.5)
    cmax = 0.1

    numroisincluster = np.nan * np.ones((len(uniquelabels),))

    # --- Compute global ylim for PSTH ---
    global_min = np.inf
    global_max = -np.inf

    for c, cluster in enumerate(uniquelabels):
        for k, tempkey in enumerate(trial_types):
            temp = populationdata[
                np.where(newlabels == cluster)[0],
                k * window_size : (k + 1) * window_size,
            ]
            if temp.size > 0:
                mean_response = np.mean(temp, axis=0)
                global_min = min(global_min, np.min(mean_response))
                global_max = max(global_max, np.max(mean_response))

            numroisincluster[c] = temp.shape[0]
            sortresponse = np.argsort(
                np.mean(temp[:, sortwindow[0] : sortwindow[1]], axis=1)
            )[::-1]
            sns.heatmap(
                temp[sortresponse],
                ax=axs[k, cluster],
                cmap=plt.get_cmap("coolwarm"),
                vmin=-cmax,
                vmax=cmax,
                cbar=(cluster == 0),
                cbar_ax=cbar_ax if (cluster == 0) else None,
                cbar_kws={"label": "Normalized fluorescence"},
            )
            axs[k, cluster].grid(False)
            axs[k, cluster].tick_params(width=0.5)
            axs[k, cluster].set_xticklabels([])
            # axs[k, cluster].set_xticklabels(
            #     [
            #         str(int((a - pre_window_size + 0.0) / framerate))
            #         for a in [
            #             0,
            #             pre_window_size,
            #             pre_window_size + frames_to_reward,
            #             window_size,
            #         ]
            #     ]
            # )
            axs[k, cluster].set_yticks([])
            axs[k, cluster].axvline(
                pre_window_size, linestyle="--", color="k", linewidth=0.5
            )
            axs[k, cluster].axvline(
                pre_window_size + frames_to_reward,
                linestyle="--",
                color="k",
                linewidth=0.5,
            )
            if cluster == 0:
                axs[k, 0].set_ylabel("%s\nNeurons" % (tempkey))

            # Plot average PSTH for each cluster, each CS
            ax = axs[-1, cluster]
            ax = tsplot(
                temp,
                ax=ax,
                color=colors_for_key[tempkey],
                label=tempkey if (cluster == (len(uniquelabels) - 1)) else None,
            )
            ax.axvline(pre_window_size, linestyle="--", color="k", linewidth=0.5)
            ax.axvline(
                pre_window_size + frames_to_reward,
                linestyle="--",
                color="k",
                linewidth=0.5,
            )
            ax.set_xticks(
                [0, pre_window_size, pre_window_size + frames_to_reward, window_size]
            )
            ax.set_xticklabels(
                [
                    str(int((a - pre_window_size + 0.0) / framerate))
                    for a in [
                        0,
                        pre_window_size,
                        pre_window_size + frames_to_reward,
                        window_size,
                    ]
                ]
            )
            
            # buffer = 0.01
            # ax.set_ylim([global_min - buffer, global_max + buffer])
            # for cluster in range(len(uniquelabels)):
            #     axs[-1, cluster].set_ylim([global_min - buffer, global_max + buffer])
            
            ax.legend(
                bbox_to_anchor=(0.94, 0.22),
                bbox_transform=fig_activity_cluster.transFigure,
                frameon=False,
            )
            standardize_plot_graphics(ax)

        axs[-1, 0].set_ylabel("Mean fluor")
        axs[0, cluster].set_title(
            "Cluster %d\n(n=%d)" % (cluster + 1, numroisincluster[c])
        )
    
    # Set average cluster activitiy ylim all the same
    buffer = 0.01
    for ax in axs[-1, :]:
        ax.set_ylim(global_min - buffer, global_max + buffer)
        
    fig_activity_cluster.text(
        0.5,
        0.05,
        "Time from cue (s)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        rotation="horizontal",
    )
    # fig_activity_cluster.tight_layout()
    fig_activity_cluster.subplots_adjust(wspace=0.1, hspace=0.1)
    fig_activity_cluster.subplots_adjust(left=0.1)
    fig_activity_cluster.subplots_adjust(right=0.93)
    fig_activity_cluster.subplots_adjust(bottom=0.1)
    fig_activity_cluster.subplots_adjust(top=0.83)

    return fig_activity_cluster


def plot_cluster_pairs(transformed_data, uniquelabels, newlabels, num_retained_pcs):

    num_clusterpairs = len(uniquelabels) * (len(uniquelabels) - 1) / 2
    colors_for_cluster = [
        [0.933, 0.250, 0.211],
        [0.941, 0.352, 0.156],
        [0.964, 0.572, 0.117],
        [0.980, 0.686, 0.250],
        [0.545, 0.772, 0.247],
        [0.215, 0.701, 0.290],
        [0, 0.576, 0.270],
        [0, 0.650, 0.611],
        [0.145, 0.662, 0.878],
        [0.604, 0.055, 0.918],
    ]
    numrows = int(np.ceil(num_clusterpairs**0.5))
    numcols = int(np.ceil(num_clusterpairs / np.ceil(num_clusterpairs**0.5)))
    fig_cluster_pairs, axs = plt.subplots(
        numrows, numcols, figsize=(3 * numrows, 3 * numcols)
    )

    tempsum = 0
    for c1, cluster1 in enumerate(uniquelabels):
        for c2, cluster2 in enumerate(uniquelabels):
            if cluster1 >= cluster2:
                continue
            temp1 = transformed_data[
                np.where(newlabels == cluster1)[0], :num_retained_pcs
            ]
            temp2 = transformed_data[
                np.where(newlabels == cluster2)[0], :num_retained_pcs
            ]
            X = np.concatenate((temp1, temp2), axis=0)
            tsne = TSNE(
                n_components=2, init="random", random_state=0, perplexity=100
            )  # visualize clusters
            Y = tsne.fit_transform(X)
            ax = axs[
                int(np.floor(tempsum / numcols)), int(np.remainder(tempsum, numcols))
            ]
            ax.scatter(
                Y[: np.sum(newlabels == cluster1), 0],
                Y[: np.sum(newlabels == cluster1), 1],
                color=colors_for_cluster[cluster1],
                label="Cluster %d" % (cluster1 + 1),
                alpha=1,
            )
            ax.scatter(
                Y[np.sum(newlabels == cluster1) :, 0],
                Y[np.sum(newlabels == cluster1) :, 1],
                color=colors_for_cluster[cluster2],
                label="Cluster %d" % (cluster2 + 1),
                alpha=1,
            )
            ax.set_xlabel("tsne dimension 1")
            ax.set_ylabel("tsne dimension 2")
            ax.legend()
            tempsum += 1
    fig_cluster_pairs.tight_layout()

    return fig_cluster_pairs


def plot_decoding_accuracy_across_time(accuracy, accuracy_chance, time_labels=None, correction='bonferroni', alpha=0.05):    
    
    # calculate mean and sem averaging animals
    mean_real = np.nanmean(accuracy, axis=0)
    sem_real = np.nanstd(accuracy, axis=0) / np.sqrt(accuracy.shape[0])
    mean_chance = np.nanmean(accuracy_chance, axis=0)
    sem_chance = np.nanstd(accuracy_chance, axis=0) / np.sqrt(accuracy_chance.shape[0])

    # Paired t-test for each timepoint
    p_values = np.array([stats.ttest_rel(accuracy[:, t], accuracy_chance[:, t], nan_policy='omit').pvalue
                         for t in range(accuracy.shape[1])])

    # Multiple comparisons correction
    reject, corrected_pvals, _, _ = multipletests(p_values, alpha=alpha, method=correction)
    significance = ['*' if r else '' for r in reject]   

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
    # title_str = f"{decoding_pair[0]} vs {decoding_pair[1]} decoding"
    # ax.set_title(title_str, fontsize=12)
    fig.tight_layout()
    
    return fig


def plot_activity_clusters_pooled_multiday(
    populationdata,
    labels,
    uniquelabels,
    numdays,
    trial_types,
    sortwindow,
    window_size,
    pre_window_size,
    frames_to_reward,
    framerate,
    day_labels: list,
    reference_day=0,
    cmax=0.1,
    percent_keep=0.30,   # <-- NEW (0-1], e.g. 0.30 for top 30%
):
    colors_for_key = {"CS1+": (0, 0.5, 1), "CS2+": (1, 0.5, 0), "CS3-": (0.5, 0.5, 0.5)}

    # clamp percent_keep
    if percent_keep is None:
        percent_keep = 1.0
    percent_keep = float(percent_keep)
    percent_keep = max(0.0, min(1.0, percent_keep))

    n_clusters = len(uniquelabels)
    n_cols = n_clusters * numdays
    n_rows = len(trial_types) + 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), squeeze=False)

    cbar_ax = fig.add_axes([0.94, 0.3, 0.01, 0.4])
    cbar_ax.tick_params(width=0.5)

    global_min, global_max = np.inf, -np.inf

    def col_of(c_idx, day):
        return c_idx * numdays + day

    for c_idx, cluster in enumerate(uniquelabels):
        idx = np.where(labels == cluster)[0]
        if idx.size == 0:
            continue

        # sorting order based on reference_day and first trial type, only among cells that exist there
        ref_exists = ~np.isnan(populationdata[idx, reference_day, 0])
        idx_ref = idx[ref_exists]
        if idx_ref.size == 0:
            continue

        ref = populationdata[idx_ref, reference_day, 0*window_size:(0+1)*window_size]
        sortresponse = np.argsort(np.mean(ref[:, sortwindow[0]:sortwindow[1]], axis=1))[::-1]
        idx_sorted = idx_ref[sortresponse]

        # --- NEW: keep only the top X% of cells in this cluster (based on idx_sorted) ---
        if percent_keep < 1.0:
            n_keep = max(1, int(np.ceil(percent_keep * idx_sorted.size)))
            idx_sorted = idx_sorted[:n_keep]
        # ---------------------------------------------------------------------------

        for day in range(numdays):
            col = col_of(c_idx, day)

            # only keep cells that exist in this day (not NaN)
            day_exists = ~np.isnan(populationdata[idx_sorted, day, 0])
            idx_day = idx_sorted[day_exists]

            for k, tempkey in enumerate(trial_types):
                temp = populationdata[idx_day, day, k * window_size:(k + 1) * window_size]

                if temp.size > 0:
                    mean_response = np.mean(temp, axis=0)
                    global_min = min(global_min, float(np.min(mean_response)))
                    global_max = max(global_max, float(np.max(mean_response)))

                sns.heatmap(
                    temp,
                    ax=axs[k, col],
                    cmap=plt.get_cmap("coolwarm"),
                    vmin=-cmax,
                    vmax=cmax,
                    cbar=(c_idx == 0 and day == 0),
                    cbar_ax=cbar_ax if (c_idx == 0 and day == 0) else None,
                    cbar_kws={"label": "Normalized fluorescence"},
                )
                axs[k, col].grid(False)
                axs[k, col].tick_params(width=0.5)
                axs[k, col].set_xticklabels([])
                axs[k, col].set_yticks([])
                axs[k, col].axvline(pre_window_size, linestyle="--", color="k", linewidth=0.5)
                axs[k, col].axvline(pre_window_size + frames_to_reward, linestyle="--", color="k", linewidth=0.5)

                if col == 0:
                    axs[k, col].set_ylabel(f"{tempkey}\nNeurons")

                axp = axs[-1, col]
                axp = tsplot(
                    temp,
                    ax=axp,
                    color=colors_for_key.get(tempkey, (0, 0, 0)),
                    label=tempkey if (c_idx == n_clusters - 1 and day == numdays - 1) else None,
                )
                axp.axvline(pre_window_size, linestyle="--", color="k", linewidth=0.5)
                axp.axvline(pre_window_size + frames_to_reward, linestyle="--", color="k", linewidth=0.5)
                axp.set_xticks([0, pre_window_size, pre_window_size + frames_to_reward, window_size])
                axp.set_xticklabels(
                    [str(int((a - pre_window_size) / framerate)) for a in
                     [0, pre_window_size, pre_window_size + frames_to_reward, window_size]]
                )
                standardize_plot_graphics(axp)

            # show n after filtering + availability on that day
            axs[0, col].set_title(
                f"Cluster {int(cluster)+1}\n {day_labels[day]}\n(n={len(idx_day)})"
            )

    if np.isfinite(global_min) and np.isfinite(global_max):
        buffer = 0.01
        for col in range(n_cols):
            axs[-1, col].set_ylim(global_min - buffer, global_max + buffer)

    axs[-1, 0].set_ylabel("Mean fluor")
    axs[-1, -1].legend(
        bbox_to_anchor=(0.94, 0.22),
        bbox_transform=fig.transFigure,
        frameon=False,
    )

    fig.text(0.5, 0.05, "Time from cue (s)", fontsize=12, ha="center", va="center")
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.1, top=0.86, wspace=0.1, hspace=0.1)

    return fig


### HERE BELOW ARE BEHAVIORAL PLOTTING FUNCTIONS
def convert_pvalue(pvalues):
    psymbols = []
    for p in pvalues:
        if p <= 0.0001:
            psymbols.append("****")
        elif p <= 0.001:
            psymbols.append("***")     
        elif p <= 0.01:
            psymbols.append("**")
        elif p <= 0.05:
            psymbols.append("*")
        else:
            psymbols.append("ns")
    return psymbols


def plot_evolution_over_learning(
    data, cue_types, colors, numdays, pvals=None, key="Performance_to_baseline"
):
    fig, axs = plt.subplots(1, 3, figsize=(2 * 3, 2), dpi=200, sharey="row")
    evolution_data = []
    mean_licks_on_day = np.nan * np.ones((numdays, 2, len(cue_types)))  # mean, sem
    animals = list(set(data["Animal"]))
    # animals_to_remove = []  # ['OFC-VTAinact_16']#['mOFCinact_32']
    # animals = [a for a in animals if a not in animals_to_remove]
    animal_num = len(animals) 
    # print("%d animals in %s" % (len(animals), condition))

    for day in range(numdays):
        nlicks = np.nan * np.ones((len(animals), len(cue_types)))
        for ct, cue_type in enumerate(cue_types):
            for a, animal in enumerate(animals):
                temp = data[
                    (data["Cue"] == cue_type)
                    & (data["Day"] == str(day + 1))
                    & (data["Animal"] == animal)
                ][key]
                if temp.size > 0:
                    nlicks[a, ct] = temp

            temp = nlicks[:, ct]
            # temp = temp[np.isfinite(temp)]
            mean_licks_on_day[day, 0, ct] = np.nanmean(temp)
            mean_licks_on_day[day, 1, ct] = stats.sem(temp, nan_policy="omit")

    if axs is not None:
        for ct, cue_type in enumerate(cue_types):
            ax = axs[ct]
            ax.errorbar(
                range(numdays),
                mean_licks_on_day[:, 0, ct],
                mean_licks_on_day[:, 1, ct],
                color=colors['ave'],
                linestyle="-",
                linewidth=1,
            )
            ax.set_xticks(range(numdays))
            ax.set_xticklabels([str(a + 1) for a in range(numdays)], fontsize=8)
            standardize_plot_graphics(ax)
        axs[1].set_xlabel("Session number", fontsize=10)
    evolution_data.append(mean_licks_on_day)

    if axs is not None:
        ax = axs[0]
        ax.set_ylabel("Mean behavioral\nperformance", fontsize=10)
        axs[-1].annotate(
            s="(n=%d)" % (animal_num),
            xy=(0.4, 0.9),
            xytext=(0.4, 0.9),
            xycoords="axes fraction",
            textcoords="axes fraction",
            color=colors['ave'],
            fontsize=7,
            horizontalalignment="left",
        )
        #         ax.set_ylim([-0.5, 2])
        
    if pvals is not None:
        psymbols = convert_pvalue(pvals)
        y1 = evolution_data[0][-1, 0, 0]
        y2 = evolution_data[1][-1, 0, 0]
        y_pos = (y1 + y2) / 2 
        axs[0].text(day + 0.3, y_pos, psymbols[0], fontsize=8, va='center')
        y1 = evolution_data[0][-1, 0, 1]
        y2 = evolution_data[1][-1, 0, 1]
        y_pos = (y1 + y2) /2
        axs[1].text(day + 0.3, y_pos, psymbols[1], fontsize=8, va='center')
        if len(psymbols) > 2:
            y1 = evolution_data[0][-1, 0, 2]
            y2 = evolution_data[1][-1, 0, 0]
            y_pos = (y1 + y2) /2
            axs[2].text(day + 0.3, y_pos, psymbols[2], fontsize=8, va='center')
    fig.tight_layout()
    return fig, evolution_data


def plot_cumlick_tbt(data, cue_types, colors, numdays, numtrials):
    fig_cumlick, axs = plt.subplots(1, 3, figsize=(6,2), dpi=200, sharey='row') # one column for each cue
    for ct, cue_type in enumerate(cue_types):
        cumlick_animals = []
        animals = list(set(data.keys())) #get animals 
        for a, animal in enumerate(animals): #for each animal
            y = data[animal][cue_type]
            # x = np.arange(1./(len(y)), 1+1./(len(y)), 1./(len(y)))
            x = np.arange(0, (len(y)), 1)
            if len(x) > len(y):
                x = x[:-1]
            cumlick_animals.append(list(y))
            ax = axs[ct]
            ax.plot(x, y, color=colors['ind'], linestyle='-', linewidth=0.5)
#             ax.axvline(float(lickidx)/(len(y)), linestyle='--', linewidth=0.5, color=colors)
            ax.set_title(cue_type, fontsize=10)
            if a==len(animals)-1:
                cumlick_animals_ave = [sum(col) / float(len(col)) for col in zip(*cumlick_animals)]
                x1 = np.arange(0, len(cumlick_animals_ave), 1)
                # x1 = np.arange(1./len(cumlick_animals_ave), 1+1./len(cumlick_animals_ave), 1./len(cumlick_animals_ave))
                ax.plot(x1, cumlick_animals_ave, color=colors['ave'], linestyle='-', linewidth=1)
                standardize_plot_graphics(ax)

        ax = axs[0]
        ax.set_ylabel('Cumulative \nanticipatory licking',fontsize=10)
        axs[1].set_xlabel('Trials',fontsize=10)
        # axs[-1].annotate(s=condition, xy=(0.4, 0.9-0.07*c), xytext=(0.4, 0.9-0.07*c),
        #                 xycoords='axes fraction', textcoords='axes fraction',
        #                 color=colors, fontsize=7,
        #                 horizontalalignment='left')
        fig_cumlick.tight_layout()
    return fig_cumlick


def plot_individual_animal_cumlick(data, learnedtrial, cue_types, colors):
    all_animals = 0
    animals = list(set(data.keys()))
    all_animals += len(animals)
    fig_ind_animals, axs = plt.subplots(all_animals, len(cue_types), figsize=(2*len(cue_types), (all_animals)), sharey='row')

    # correct_trial = 0
    # trial_to_end = 0

    for a, animal in enumerate(animals): #for each animal
#         print(animal)
        for ct, cue_type in enumerate(cue_types):
            y = data[animal][cue_type]
            x = np.arange(0, (len(y)), 1)
            ax = axs[a, ct]
            ax.plot(x, y, color=colors['ind'], linestyle='-', linewidth=1)
            ax.plot([x[0], x[-1]], [y[0], y[-1]], linestyle='--', color='#808080')
            if ct != 2:
                learned_trial = learnedtrial[cue_type][animal]
                ax.axvline(learned_trial, linestyle='--', linewidth=0.5, color=colors['ave'])
            standardize_plot_graphics(ax)
        ax = axs[a, 0]
        ax.set_ylabel('Anticipatory\nlicking',fontsize=8)
        ax.set_title(animal, fontsize=5, loc='left')
    ax2 = axs[-1,0]
    ax2.set_xlabel('Trials',fontsize=8)
    ax3 = axs[-1,1]
    ax3.set_xlabel('Trials',fontsize=8)
    fig_ind_animals.tight_layout()

    return fig_ind_animals


def plot_changepoint(data, cue_types, colors, param:str):
     # ylabel
    if param == "abruptness":
        ylabel = "Abruptness of\nlearning"
    elif param == "learned trial":
        ylabel = "Learned trial"
    elif param == "mean slope after learning":
        ylabel = "Mean slope\nafter learning"
    else:
        ylabel = param

    fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=200)

    # --- collect all animals across cue types ---
    animals = sorted({a for ct in cue_types for a in data.get(ct, {}).keys()})

    x = np.arange(len(cue_types))

    # --- plot per-animal lines ---
    for animal in animals:
        y = []
        for ct in cue_types:
            v = data.get(ct, {}).get(animal, np.nan)

            # if v is array-like (e.g., list), try to reduce to a scalar
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v)
                v = np.nanmean(v) if v.size else np.nan

            y.append(v)

        y = np.asarray(y, dtype=float)
        ax.plot(
            x, y,
            color=colors.get("ind", "0.7"),
            linewidth=0.6,
            marker="o",
            markersize=2,
        )

    # --- overlay mean ± SEM per cue type ---
    means = np.full(len(cue_types), np.nan, dtype=float)
    sems  = np.full(len(cue_types), np.nan, dtype=float)

    for i, ct in enumerate(cue_types):
        vals = np.array(list(data.get(ct, {}).values()), dtype=float)

        # if some entries are arrays, reduce them
        if vals.dtype == object:
            vv = []
            for v in data.get(ct, {}).values():
                if isinstance(v, (list, tuple, np.ndarray)):
                    v = np.asarray(v)
                    v = np.nanmean(v) if v.size else np.nan
                vv.append(v)
            vals = np.asarray(vv, dtype=float)

        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            means[i] = np.mean(vals)
            sems[i] = np.std(vals) / math.sqrt(vals.size)

    ax.errorbar(
        x, means, yerr=sems,
        color=colors.get("ave", "k"),
        linewidth=1.2,
        marker="o",
        markersize=3,
        capsize=3,
        zorder=5
    )

    # --- cosmetics ---
    ax.set_xticks(x)
    ax.set_xticklabels(cue_types, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(param, fontsize=10)

    # optional: tighten x-limits a bit
    ax.set_xlim(-0.3, len(cue_types) - 0.7)
    ax.set_ylim(0, np.max(vals) * 1.2)

    standardize_plot_graphics(ax)
    fig.tight_layout()
    return fig


def plot_ave_psth_overlay_days(
    psth_df,
    cues=("CS1", "CS2", "CS3"),
    colors=None,                 # e.g. {"ave": (0,0.5,1), "ind": (0,0.5,1)} or similar
    row_mode="Day",              # "Day" or "Session"
    row_values=None,             # list of days/sessions to overlay (sorted = earliest -> latest)
    animal_weighted=True,
    smooth_bins=0,
    hz=True,
    show_sem=True,
    sem_alpha=0.15,              # base alpha for SEM; will also be scaled by day alpha
    alpha_range=(0.25, 1.0),     # earliest -> latest darkness
):
    df = psth_df.copy()
    row_col = row_mode

    if row_values is None:
        row_values = sorted(df[row_col].dropna().unique().tolist())

    if colors is None:
        colors = {"ave": "C0", "ind": "C0"}  # fallback

    bin_ms = float(df["Bin_ms"].iloc[0])
    to_hz = (bin_ms / 1000.0) if hz else 1.0

    ncols = len(cues)
    fig, axs = plt.subplots(
        1, ncols,
        figsize=(3.8 * ncols, 2.8),
        dpi=200,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axs = axs[0]

    def smooth(x, k):
        if k and k > 1:
            k = int(k)
            ker = np.ones(k) / k
            return np.convolve(x, ker, mode="same")
        return x

    # alpha schedule: earliest lightest -> latest darkest
    if len(row_values) == 1:
        alphas = [alpha_range[1]]
    else:
        alphas = np.linspace(alpha_range[0], alpha_range[1], len(row_values))

    base_line_rgba = to_rgba(colors.get("ave", "C0"))
    base_fill_rgba = to_rgba(colors.get("ind", colors.get("ave", "C0")))

    for c, cue in enumerate(cues):
        ax = axs[c]

        for rv, a in zip(row_values, alphas):
            sub = df[(df[row_col] == rv) & (df["Cue"] == cue)]
            if sub.empty:
                continue

            if animal_weighted:
                at = sub.groupby(["Animal", "Time_s"], as_index=False)["Count"].mean()
                piv = at.pivot(index="Animal", columns="Time_s", values="Count")
                mat = piv.to_numpy(dtype=float)
                t = piv.columns.to_numpy(dtype=float)
            else:
                tt = sub.groupby(["Trial", "Time_s"], as_index=False)["Count"].mean()
                piv = tt.pivot(index="Trial", columns="Time_s", values="Count")
                mat = piv.to_numpy(dtype=float)
                t = piv.columns.to_numpy(dtype=float)

            mean = np.nanmean(mat, axis=0) / to_hz
            if smooth_bins:
                mean = smooth(mean, smooth_bins)

            if mat.shape[0] > 1:
                sem = (np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])) / to_hz
            else:
                sem = np.zeros_like(mean)

            if smooth_bins:
                sem = smooth(sem, smooth_bins)

            line_rgba = (*base_line_rgba[:3], a)
            fill_rgba = (*base_fill_rgba[:3], min(1.0, sem_alpha * a))

            ax.plot(t, mean, color=line_rgba, lw=2.0, label=f"{row_col} {rv}")
            if show_sem:
                ax.fill_between(t, mean - sem, mean + sem, color=fill_rgba, lw=0)

        # event lines
        ax.axvline(0, color="k", linestyle="--", lw=1)
        ax.axvline(3, color="orange", linestyle="--", lw=1)

        ax.set_title(cue)
        ax.set_xlabel("Time from cue (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if c == 0:
            ax.set_ylabel("Lick rate (Hz)" if hz else "Count/bin")

        # put legend on last cue axis
        if c == ncols - 1:
            ax.legend(frameon=False, fontsize=8, loc="upper right", title=row_col)

    fig.tight_layout()
    return fig, axs
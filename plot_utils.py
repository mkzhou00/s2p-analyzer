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

sns.set_style("ticks")
import matplotlib as mpl

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

    for c, cluster in enumerate(uniquelabels):
        for k, tempkey in enumerate(trial_types):
            temp = populationdata[
                np.where(newlabels == cluster)[0],
                k * window_size : (k + 1) * window_size,
            ]
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
            if cluster == 0:
                ax.set_ylim([-0.02, 0.02])
            else:
                ax.set_yticks([])
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

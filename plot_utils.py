import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['legend.labelspacing'] = 0.2
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['ytick.major.pad'] = 1
mpl.rcParams['lines.scale_dashes'] = False
mpl.rcParams['lines.dashed_pattern'] = (2, 1)
mpl.rcParams['font.sans-serif'] = ['Helvetica LT Std']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['text.color'] = 'k'


def plot_raw_licks(CS, licks, before, after):
    fig_rawplot, ax = plt.subplots(1, len(CS), figsize=(3*(len(CS)), 3))
    if ax is None:
        fig, ax = plt.subplots(1,1)
    for cs_type, (cs, a) in enumerate(zip(CS, ax)):
        if cs_type == 2:
            cs_sign = '-'
        else:
            cs_sign = '+'
        for i in range(0, len(cs)):
            raw_licks = licks[(licks >= ((cs[i]) - before)) & (licks <= (cs[i] + after))]
            a.vlines(x=(raw_licks-cs[i]), ymin=i, ymax=(i+1), linewidth=1, color='#654321')
            if i==len(cs)-1:
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
                a.set_xlim(-before, after)
                a.set_ylim(0,len(cs))
                a.vlines(x=0, ymin=0, ymax=len(cs), linestyles='dashed',color='k', linewidth=1)
                a.vlines(x=3, ymin=0, ymax=len(cs), linestyles='dashed',color='k', linewidth=1)
                a.set_xlabel('Time (s)')
                a.set_title(('CS'+str(cs_type+1)+cs_sign))
        ax[0].set_ylabel('Trials')
    return fig_rawplot


def plot_average_PSTH_around_interest_window(CS: list, Fcorr):
    fig_PSTH, axs = plt.subplots(1, len(CS), figsize=(4*(len(CS)), 4))
    # ax = axs[0]
    # sns.heatmap(cs[sortresponse,:],
    #             ax=ax,
    #             cmap=plt.get_cmap('coolwarm'),
    #             vmax=0.1, vmin=-0.1)
    # ax.set_title('CS+')
    temp = []
    for cue_type in range(len(CS)):
        if cue_type == 2:
            cs_sign = '-'
        else:
            sc_sign = '+'
        ax = axs[cue_type]
        sortresponse = np.argsort(np.mean(Fcorr[cue_type], axis=1))
        sns.heatmap(Fcorr[cue_type][sortresponse],
                ax=ax,
                cmap=plt.get_cmap('coolwarm'),
                vmax=0.1, vmin=-0.1)
        ax.set_title(('CS'+str(cue_type+1)+cs_sign))   
        ax.grid(False)
        ax.set_ylabel('Neurons')
        ax.set_xlabel('Time from cue (s)')
        # ax.set_yticks(list(range(0,populationdata.shape[0],500)))
        # ax.set_yticklabels([str(a+1) for a in range(0,populationdata.shape[0],500)])
        ax.set_xticks([-3, 1, 10])
        # ax.set_xticklabels([str(int((a-pre_window_size+0.0)/framerate))
        #                                         for a in [0, pre_window_size,
        #                                                 pre_window_size + frames_to_reward, window_size]])
        # ax.axvline(pre_window_size, linestyle='--', color='k', linewidth=0.5) 
        # ax.axvline(pre_window_size + frames_to_reward, linestyle='--', color='k', linewidth=0.5)
    fig_PSTH.tight_layout()        

    return fig_PSTH


def plot_correlation_martix(CS, Forr):
    sortresponse = np.argsort(np.mean(cs[:,pre_window_size:pre_window_size+frames_to_reward], axis=1))[::-1]
    fig, axs = plt.subplots(1,2, figsize=(8,4))



    
def plot_average_PSTH_around_event(cs, framenumberforevent, framerate, frames_to_reward, savedir,
                                   window_size=30, pre_window_size=10, trialsofinterest=None,
                                   sortby='response', eventname='first lick after unpredicted reward',
                                   centraltendency='baseline subtracted mean'):

    sortresponse = np.argsort(np.mean(cs[:,pre_window_size:pre_window_size+frames_to_reward], axis=1))[::-1]
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    ax = axs[0]
    sns.heatmap(cs[sortresponse,:],
                ax=ax,
                cmap=plt.get_cmap('coolwarm'),
                vmax=0.1, vmin=-0.1)
    ax.set_title('CS+')

    for ax in axs:
        ax.grid(False)
        ax.set_ylabel('Neurons')
        ax.set_xlabel('Time from cue (s)')
        ax.set_yticks(list(range(0,populationdata.shape[0],500)))
        ax.set_yticklabels([str(a+1) for a in range(0,populationdata.shape[0],500)])
        ax.set_xticks([0, pre_window_size,pre_window_size + frames_to_reward, window_size])
        ax.set_xticklabels([str(int((a-pre_window_size+0.0)/framerate))
                                                for a in [0, pre_window_size,
                                                        pre_window_size + frames_to_reward, window_size]])
        ax.axvline(pre_window_size, linestyle='--', color='k', linewidth=0.5) 
        ax.axvline(pre_window_size + frames_to_reward, linestyle='--', color='k', linewidth=0.5)
    fig.tight_layout()
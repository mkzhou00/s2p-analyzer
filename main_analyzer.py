import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    plot_correlation_martix,
)


logger = logging.getLogger(__name__)


def parse_args():
    """Parses arguments from command line."""
    parser = argparse.ArgumentParser(description="Suite2p result analyzer.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="/Users/mzhou/Library/CloudStorage/OneDrive-UCSF/MZ_hpc_prism_M4/d5/",
        help="Directory containing all the required files.",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="result/",
        help="folder for saving figures and results",
    )
    parser.add_argument(
        "--num_planes",
        type=int,
        default="2",
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
        default=10,
        help="Interested time region after a cue starts. (Second)",
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
        default=3,
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
    args = parse_args()
    data_loader = DataLoader(args.data_dir, args.num_planes)

    # Load all necessary data.
    F = data_loader.get_F()
    Fneu = data_loader.get_Fneu()
    stat = data_loader.get_stat()
    is_cell = data_loader.get_is_cell()
    # ops = data_loader.get_ops()
    # spks = data_loader.get_spks()
    event_df = data_loader.get_event_df()  # Arduino
    voltages = data_loader.get_voltages()  # Computer
    im_ts = data_loader.get_im_ts()  # image time stamps in second

    if args.num_planes > 1:
        overlapping_cells = correct_overlapping_cells_across_planes(stat, is_cell)

    # `event_tf` contains Arduino timestamps.
    event_cues, new_event_cues, voltage_cues = correct_arduino_timestamps(
        event_df, voltages
    )

    # Make a result folder
    result_dir = os.path.join(args.data_dir, args.result_folder)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Get F_cell and Fneu_cell only activity
    F_cell, Fneu_cell = get_cell_only_activity(
        F, Fneu, is_cell, args.num_planes, args.cell_threshold
    )
    # Get neuropil corrected F with neuropil coefficient
    assert len(F_cell) == len(Fneu_cell), "Fcell and Fneu cell must be the same length"
    Fcorr = get_corrected_F(F_cell, Fneu_cell, args.num_planes, args.neucoeff)
    for ip in range(args.num_planes):
        assert len(F_cell[ip]) == len(Fcorr[ip])
    Fcorr = normalize_signal(Fcorr, args.num_planes, 'z_score')

    # Extract all event time points from new event_df
    [licks, CS1, CS2, CS3, sucrose, milk] = extract_events(event_df)
    allCS = [CS1, CS2, CS3]

    ## Plot behavior rasters
    fig_rawplot = plot_raw_licks(
        allCS, licks, args.pre_cue_window, args.post_cue_window
    )
    plt.close(fig_rawplot)
    fig_rawplot.savefig(os.path.join(result_dir, 'behavior_raster.png'), format='png')

    # Extract time around each cue and sorted by CS type
    interest_intervals = extract_interest_time_intervals(allCS, args)
    # Extract image time points around each cue and sorted by CS type and plane
    im_idx_around_cue = extract_imaging_ts_around_events(
        allCS, im_ts, args.num_planes, interest_intervals
    )
    # Extract average Fcorr around each cue in all cuetypes for each cell
    Fcorr_around_cue, Fcorr_around_cue_baseline_subtract = extract_Fave_around_events(
        allCS, Fcorr, im_idx_around_cue, args.num_planes, args.pre_cue_window
    )
    # cells_to_plot = [1, 3, 4, 5, 6, 8, 10]
    # for ip in range(args.num_planes):
    #     cells = np.argsort(np.max(Fcorr[ip], axis=1))[::-1][0:10]
    #     cells_to_plot.append(cells)
    # plot_individual_cells_activity(Fcorr, allCS, im_idx_around_cue, cells_to_plot, args.num_planes)

    # fig_calcium_PSTH = plot_average_PSTH_around_interest_window(allCS, Fcorr_around_cue,'median')
    fig_calcium_PSTH, cells_sort_by_activity = plot_average_PSTH_around_interest_window(allCS, Fcorr_around_cue_baseline_subtract)
    fig_calcium_PSTH.savefig(os.path.join(result_dir, 'PSTH_heatmap_zscore.png'), format='png')
    plt.close(fig_calcium_PSTH)
    cells_to_plot = cells_sort_by_activity[0:int(np.floor(0.1*len(cells_sort_by_activity)))]
    # cells_to_plot = list(range(len(Fcorr[0]),  len(Fcorr[0])+30))
    plot_individual_cells_activity(Fcorr, allCS, im_idx_around_cue, cells_to_plot, args.num_planes, result_dir)   
    # correlationmatrix = plot_correlation_martix(allCS, Fcorr)


if __name__ == "__main__":
    main()

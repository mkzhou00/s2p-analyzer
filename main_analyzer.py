import argparse
import logging
import pandas as pd
import numpy as np

from s2p_utils.data_loader import DataLoader
from s2p_utils.processing_utils import (
    correct_arduino_timestamps,
    extract_cues_from_events,
    get_cell_only_activity,
)


logger = logging.getLogger(__name__)


def parse_args():
    """Parses arguments from command line."""
    parser = argparse.ArgumentParser(description="Suite2p result analyzer.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="/Users/mzhou/Library/CloudStorage/OneDrive-UCSF/PhD projects/analysis/sample_for_analysis/full_session/",
        help="Directory containing all the required files.",
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
        "--event_interest_time_region_before",
        type=float,
        default=3,
        help="Interested time region before a cue starts. (Second)",
    )
    parser.add_argument(
        "--event_interest_time_region_after",
        type=float,
        default=3,
        help="Interested time region after a cue starts. (Seoncd)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=".",
        help="Output Folder.",
    )
    return parser.parse_args()


def extract_interest_time_intervals(event_df: pd.DataFrame, args):
    event_cues = extract_cues_from_events(event_df)
    interest_interval = []

    for cue in event_cues["Timestamp"]:
        cue_timestamp_s = cue / 1000  # Convert from ms to s.
        interest_interval.append(
            [
                cue_timestamp_s - args.event_interest_time_region_before,
                cue_timestamp_s + args.event_interest_time_region_after,
            ]
        )
    return interest_interval


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
    # Fneu corrected by suite2p
    stat = data_loader.get_stat()
    is_cell = data_loader.get_is_cell()
    ops = data_loader.get_ops()
    spks = data_loader.get_spks()
    event_df = data_loader.get_event_df()  # Arduino
    voltages = data_loader.get_voltages()  # Computer
    im_ts = data_loader.get_im_ts() # image time stamps in second

    # `event_tf` contains Arduino timestamps.
    event_cues, new_event_cues, voltage_cues = correct_arduino_timestamps(event_df, voltages)

    # Making sure they are the same size for `zip()`
    for ip in range(args.num_planes):
        assert len(is_cell[ip]) == len(F[ip]) and len(F[ip]) == len(
            stat[ip]
        ), "F and is_cell must be of the same length."

    # Extract time intervals of interest.
    interest_interval = extract_interest_time_intervals(event_df, args)

    # Variables holding associated cell data.
    # Key: cell index, Val: set of intervals in which this cell's activity is detected.
    cells_active_before_cue = {}
    cells_active_after_cue = {}
    # Key: interval index, Val: Set of cells' index detected in the interval
    intervals_before_cue_cells = {}
    intervals_after_cue_cells = {}
    F_cell_trials = {}
    spks_cell_trials = {}


    # Get traces and spikes for ROIs identified as cells
    F_cell = get_cell_only_activity(F, is_cell, args.num_planes)
    spks_cell = get_cell_only_activity(spks, is_cell, args.num_planes)

    for ip in range(args.num_planes):
        for ic, (F_ic, spk_ic) in enumerate(zip(F_cell[ip], spks_cell[ip])):
            # Skip if cell probability is low.
            # if is_cell_i[1] < args.min_cell_prob:
            #     continue

            # "F_ic is each cell's raw trace activities, spk_ic is each cell's spike activities."
            for interval_idx, (start, end) in enumerate(interest_interval):
                for itime, (time, activity) in enumerate(zip(im_ts[ip], F_ic, spk_ic)):
                    if (time >= start & time <= end):
                        F_cell_trials[ip][ic].add()
                        

                #         if i not in cells_active_before_cue[ip]:
                #             cells_active_before_cue[ip][i] = set()  # no repeating elements
                #         cells_active_before_cue[ip][i].add(interval_idx)


                #         if interval_idx not in intervals_before_cue_cells[ip]:
                #             intervals_before_cue_cells[ip][interval_idx] = set()
                #         intervals_before_cue_cells[ip][interval_idx].add(i)

                # for interval_idx, (start, end) in enumerate(
                #     interest_interval_after_cue
                # ):
                #     if (time >= start and time <= end):
                #         if i not in cells_active_after_cue[ip]:
                #             cells_active_after_cue[ip][i] = set()
                #         cells_active_after_cue[ip][i].add(interval_idx)

                #         if interval_idx not in intervals_after_cue_cells[ip]:
                #             intervals_after_cue_cells[ip][interval_idx] = set()
                #         intervals_after_cue_cells[ip][interval_idx].add(i)


if __name__ == "__main__":
    main()

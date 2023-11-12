import argparse
import logging
import pandas as pd
import numpy as np

from s2p_utils.data_loader import DataLoader
from s2p_utils.processing_utils import (
    correct_arduino_timestamps,
    extract_cues_from_events,
)


logger = logging.getLogger(__name__)


def parse_args():
    """Parses arguments from command line."""
    parser = argparse.ArgumentParser(description="Suite2p result analyzer.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing all the required files.",
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
    interest_interval_before_cue = []
    interest_interval_after_cue = []
    for cue in event_cues["Timestamp"]:
        cue_timestamp_s = cue / 1000  # Convert from ms to s.
        interest_interval_before_cue.append(
            [
                cue_timestamp_s - args.event_interest_time_region_before,
                cue_timestamp_s,
            ]
        )
        interest_interval_after_cue.append(
            [
                cue_timestamp_s,
                cue_timestamp_s + args.event_interest_time_region_after,
            ]
        )
    return interest_interval_before_cue, interest_interval_after_cue


def associate_cells_with_intervals(
    interval: list,
    time: float,
    cell_indices_with_activities: dict[int, set],
    intervals_with_detected_cells: dict[int, set],
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
    data_loader = DataLoader(args.data_dir)

    # Load all necessary data.
    F = data_loader.get_F()
    stat = data_loader.get_stat()
    is_cell = data_loader.get_is_cell()
    event_df = data_loader.get_event_df()  # Arduino
    voltages = data_loader.get_voltages()  # Computer

    # `event_tf` contains Arduino timestamps.
    correct_arduino_timestamps(event_df, voltages)

    # Making sure they are the same size for `zip()`
    assert len(is_cell) == len(F) and len(F) == len(
        stat
    ), "F and is_cell must be of the same length."

    # Extract time intervals of interest.
    (
        interest_interval_before_cue,
        interest_interval_after_cue,
    ) = extract_interest_time_intervals(event_df, args)

    # Variables holding associated cell data.
    # Key: cell index, Val: set of intervals in which this cell's activity is detected.
    cell_activities_before_cue = {}
    cell_activities_after_cue = {}
    # Key: interval index, Val: Set of cells' index detected in the interval
    intervals_before_cue_cells = {}
    intervals_after_cue_cells = {}

    for i, (F_i, is_cell_i, stat_i) in enumerate(zip(F, is_cell, stat)):
        # Skip if cell probability is low.
        if is_cell_i[1] < args.min_cell_prob:
            continue

        # `F_i` contains timestamps(s) where cell is found.
        for time in F_i:
            for interval_idx, (start, end) in enumerate(interest_interval_before_cue):
                if time >= start and time <= end:
                    if i not in cell_activities_before_cue:
                        cell_activities_before_cue[i] = set()
                    cell_activities_before_cue[i].add(interval_idx)

                    if interval_idx not in intervals_before_cue_cells:
                        intervals_before_cue_cells[interval_idx] = set()
                    intervals_before_cue_cells[interval_idx].add(i)

            for interval_idx, (start, end) in enumerate(interest_interval_after_cue):
                if time >= start and time <= end:
                    if i not in cell_activities_after_cue:
                        cell_activities_after_cue[i] = set()
                    cell_activities_after_cue[i].add(interval_idx)

                    if interval_idx not in intervals_after_cue_cells:
                        intervals_after_cue_cells[interval_idx] = set()
                    intervals_after_cue_cells[interval_idx].add(i)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np


def correct_overlapping_cells_across_planes(stat, iscell):
    cellidx = []
    for ip in range(len(iscell)):
        # get the index number of ROIs that is cell (1 in first item of iscell)
        temp = (iscell[ip] == 1).nonzero()[0]
        cellidx.append(temp)

    xy_plane0 = set()
    overlapping_cells = []
    for icell in cellidx[0]:
        assert len(stat[0][0][icell]["xpix"][0][0][0]) == len(
            stat[0][0][icell]["ypix"][0][0][0]
        )
        xpix = stat[0][0][icell]["xpix"][0][0][0]
        ypix = stat[0][0][icell]["ypix"][0][0][0]
        reference_cell_coordinates = set()
        for ix, iy in zip(xpix, ypix):
            reference_cell_coordinates.add((ix, iy))
        for ic in cellidx[1]:
            assert len(stat[1][0][ic]["xpix"][0][0][0]) == len(
                stat[1][0][ic]["ypix"][0][0][0]
            )
            x = stat[1][0][ic]["xpix"][0][0][0]
            y = stat[1][0][ic]["ypix"][0][0][0]
            overlap_ct = 0
            for ix, iy in zip(x, y):
                if (ix, iy) in reference_cell_coordinates:
                    overlap_ct += 1
            if (overlap_ct / len(x)) >= 0.8:
                iscell[1][ic][0] = 0
                overlapping_cells.append([icell, ic])

    return overlapping_cells


def extract_cues_from_events(event_df: pd.DataFrame) -> pd.DataFrame:
    return np.array(
        event_df.loc[
            (event_df["Events"] == 15)  # CS1
            | (event_df["Events"] == 16)  # CS2
            | (event_df["Events"] == 17)  # CS3
        ]["Timestamp"]
    )


def extract_cues_from_voltages(voltages: pd.DataFrame) -> pd.DataFrame:
    """
    Return the indexes of time points for each cue --> TTL2 turns bigger than 3V.

    """
    event_voltages = np.array(voltages[" TTL2"])

    # Large voltage difference (positive) when cue starts. `diff` results in one
    # less element than the orginal array. Pad a zero in front.
    diff = np.diff(event_voltages)
    padded_diff = np.concatenate([[0], diff])

    assert len(padded_diff) == len(voltages)

    voltages["TTL2 diff"] = padded_diff
    return np.array(voltages.loc[voltages["TTL2 diff"] > 3]["Time(ms)"])


def extract_cues(event_df: pd.DataFrame, voltages: pd.DataFrame):
    """
    Extract cue times from arduino recorded events and voltages.

    """
    event_cues = extract_cues_from_events(event_df)
    voltage_cues = extract_cues_from_voltages(voltages)

    # Check thresholds set in the extract functions when the following assert
    # is triggered.
    assert len(event_cues) == len(voltage_cues), (
        "Cues extracted from events and voltages must match."
        f"Event Cues: {len(event_cues)},"
        f"voltage cues: {len(voltage_cues)}"
    )

    return event_cues, voltage_cues


def correct_arduino_timestamps(event_df: pd.DataFrame, voltages: pd.DataFrame):
    """
    Arduino time drifts (assume linear) w.r.t. computer time. Correct timestamps
    collected on Arduino given corresponding computer timestamps.

    Args:
        event_df[in/out]: Events w/ timestamps collected on Arduino.
        voltages[in]: Corresponding timestamps collected on computer.
    """
    ### Voltages (computer received)

    # Extract in-session ts between start (1) and end (0).
    v_in_session = voltages[voltages[" TTL1"] > 3]

    # Get start ts.
    v_session_first_ts = v_in_session.iloc[0][0]

    # Subtract all ts using first timestamp, effectively making event starts at 0.
    v_in_session["Time(ms)"] -= v_session_first_ts

    # Do the same thing for Events (on Arduino), setting event start at 0.
    e_session_start = event_df.loc[event_df["Events"] == 1]
    event_df["Timestamp"] = (
        event_df["Timestamp"].to_numpy() - e_session_start["Timestamp"].to_numpy()
    )

    # Now that both data starts at 0, correct the linear drift based on event cues.
    event_cues, voltage_cues = extract_cues(event_df, voltages)

    assert len(event_cues) > 0

    ## Linear scaling across time points
    # scale = 0
    # for e_cue, v_cue in zip(
    #     event_cues["Timestamp"].to_numpy(), voltage_cues["Time(ms)"].to_numpy()
    # ):
    #     scale += v_cue / e_cue
    # scale /= len(event_cues)
    # # Correct for linear scale.
    # event_df["Timestamp"] *= scale
    # new_event_cues = extract_cues_from_events(event_df)

    # for non linear scaling across time points, correct for each cue
    for icue, (e_cue, v_cue) in enumerate(zip(event_cues, voltage_cues)):
        if icue < (len(event_cues) - 1):
            scale = v_cue / e_cue
            idx_events_after_cue = event_df.loc[
                (event_df["Timestamp"] >= e_cue)
                & (event_df["Timestamp"] < (event_cues[icue + 1]))
            ].index.tolist()
            event_df["Timestamp"][idx_events_after_cue] *= scale
        elif icue == (len(event_cues) - 1):
            scale = v_cue / e_cue
            idx_events_after_cue = event_df.loc[
                (event_df["Timestamp"] >= e_cue)
            ].index.tolist()
            event_df["Timestamp"][idx_events_after_cue] *= scale
    new_event_cues = extract_cues_from_events(event_df)

    return event_cues, new_event_cues, voltage_cues


def extract_events(event_df: pd.DataFrame):
    # get all events in seconds
    licks = np.array(event_df["Timestamp"][event_df["Events"] == 5] / 1e3)
    CS1 = np.array(event_df["Timestamp"][event_df["Events"] == 15] / 1e3)
    CS2 = np.array(event_df["Timestamp"][event_df["Events"] == 16] / 1e3)
    CS3 = np.array(event_df["Timestamp"][event_df["Events"] == 17] / 1e3)
    sucrose = np.array(
        event_df["Timestamp"][(event_df["Events"] == 10) & (event_df["Reward"] == 1)]
        / 1e3
    )
    milk = np.array(
        event_df["Timestamp"][(event_df["Events"] == 9) & (event_df["Reward"] == 1)]
        / 1e3
    )
    return licks, CS1, CS2, CS3, sucrose, milk


def get_cell_only_activity(
    F: list, Fneu: list, is_cell: list, num_planes: int, threshold: int
):
    """
    Returns cell only activity for traces and spikes based on is_cell, 1==cell, 0==not cell in is_cell, for each plane.

    """
    F_cell = [[] for _ in range(num_planes)]
    Fneu_cell = [[] for _ in range(num_planes)]

    for ip in range(num_planes):
        cell_idx = [index for index, value in enumerate(is_cell[ip]) if value[0] == 1]
        for cell in cell_idx:
            if np.mean(F[ip][cell, :] / Fneu[ip][cell, :]) >= (1 + threshold / 1e3):
                F_cell[ip].append(F[ip][cell, :])
                Fneu_cell[ip].append(Fneu[ip][cell, :])

    return F_cell, Fneu_cell


def get_corrected_F(F_cell: list, Fneu_cell: list, num_planes: int, coeff: float):
    Fcorr = [[] for _ in range(num_planes)]
    for ip in range(num_planes):
        for ic, (fc, fneuc) in enumerate(zip(F_cell[ip], Fneu_cell[ip])):
            # equation for neuro pil F correction
            temp = fc - fneuc * coeff
            Fcorr[ip].append(temp)
    return Fcorr


def extract_interest_time_intervals(event_cues, args):
    """
    This function takes the event cue times and interest time window to generate interest time intervals around each cue.
    Total length is the number of cues.

    """
    # creat empty list to contain interest time window for each CS type
    interest_interval = [[] for _ in range(len(event_cues))]
    for ct, cue_type in enumerate(event_cues):
        for cue in cue_type:
            interest_interval[ct].append(
                [
                    cue - args.pre_cue_window,
                    cue + args.post_cue_window,
                ]
            )
    return interest_interval


def extract_imaging_ts_around_events(CS, im_ts, num_planes: int, interest_intervals):
    """
    This function generates image indexes around time of interest, here are CSs.

    Args:
        CS: all CSs
        im_ts: image time frames extracted from xml file
        num_planes: number of planes
        interest_intervals: interest interval around each cue in seconds

    Return:
        The image indexes around interest interval for each cue
    """

    im_idx_around_cues = [[[] for _ in range(len(CS))] for _ in range(num_planes)]
    for cs_type, cs in enumerate(CS):
        for ip in range(num_planes):
            for interval in interest_intervals[cs_type]:
                # find the images condition during each interval
                condition_idx = (im_ts[ip] >= interval[0]) & (im_ts[ip] <= interval[1])
                # get the image time points for each cue
                cue_temp = [
                    i
                    for i, (ts, condition) in enumerate(zip(im_ts[0], condition_idx))
                    if condition
                ]
                # append image time points for each cue under correct CS type and plane
                im_idx_around_cues[ip][cs_type].append(cue_temp)
    im_idx_around_cues = np.array(im_idx_around_cues)
    return im_idx_around_cues

def normalize_signal(Fcorr, num_planes: int, norm_by='median'):
    """
    This function normalizes Fcorrected traces.

    """
    if norm_by == 'z_score':
        for ip in range(num_planes):
            mean = np.nanmean(Fcorr[ip], axis=1)
            std = np.std(Fcorr[ip], axis=1)
            Fcorr[ip] = (Fcorr[ip] - mean[:,None]) / std[:,None]
    elif norm_by == 'median':
        for ip in range(num_planes):
            median = np.median(Fcorr[ip], axis=1)
            max = np.max(Fcorr[ip], axis=1)
            min = np.min(Fcorr[ip], axis=1)
            Fcorr[ip] = (Fcorr[ip] - median[:,None]) / (max[:,None] - min[:,None])
    return Fcorr


def extract_Fave_around_events(CS, F, im_idx_around_cues, num_planes: int, pre_cue_window: int):
    """
    This function first generates Fcorrected traces around each cues based on input images indexes,
    and average Fcorr across all CS trials within the same CS type for each cell,
    and append each cell's average activity under each cue.

    Args:
        CS: all CS trials
        F: Fcorrected trace for all planes all cells
        im_dx: image indexes around each cue
        num_planes: number of planes

    Returns:
    Fcorrected_around_cue with the structure of len(CS) x number of cells

    """
    F_ave_around_cues = [[] for _ in range(len(CS))]
    F_ave_around_cues_baseline_subtract = [[] for _ in range(len(CS))]

    for cue_type, cs in enumerate(CS):  # cue_type = 0,1,2 (CS1, CS2, CS3)
        for ip in range(num_planes):
            cue_ts = im_idx_around_cues[ip][
                cue_type
            ]  # image indexes for cues, holds same for all cells within the plane
            for cell in range(len(F[ip])):
                cell_F = []
                for trial in range(len(cs)):
                    F_temp = F[ip][cell][cue_ts[trial]]
                    # Initialize frame number 
                    if cue_type == 0:
                        if trial == 0:
                            framenumber = len(F_temp)
                    # Correct for frame for each trial
                    if len(F_temp) > framenumber:
                        # if images number is bigger than default, drop the extra ones
                        F_temp = F_temp[0:framenumber]
                    elif len(F_temp) < framenumber:
                        # if images is smaller than default, add nan at the end to fill the spots
                        for i in range(framenumber - len(F_temp)):
                            F_temp = np.append(F_temp, np.nan)
                    cell_F.append(F_temp)
                # average across cs trials
                cellave = np.nanmean(np.array(cell_F), axis=0)
                baseline = np.nanmean(cellave[0:pre_cue_window * 10])
                baselinesubtract = cellave - baseline
                F_ave_around_cues_baseline_subtract[cue_type].append(baselinesubtract)
                F_ave_around_cues[cue_type].append(cellave)
    return F_ave_around_cues, F_ave_around_cues_baseline_subtract
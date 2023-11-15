import pandas as pd
import numpy as np


def extract_cues_from_events(event_df: pd.DataFrame) -> pd.DataFrame:
    return event_df.loc[
        (event_df["Events"] == 15)  # CS1
        | (event_df["Events"] == 16)  # CS2
        | (event_df["Events"] == 17)  # CS3
    ]


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
    return voltages.loc[voltages["TTL2 diff"] > 3]


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

    # Averge scales at different time points.
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
    for icue, (e_cue, v_cue) in enumerate(
        zip(event_cues["Timestamp"].to_numpy(), voltage_cues["Time(ms)"].to_numpy())
    ):
        if icue < (len(event_cues) - 1):
            scale = v_cue / e_cue
            idx_events_after_cue = event_df.loc[
                (event_df["Timestamp"] >= e_cue)
                & (event_df["Timestamp"] < (event_cues["Timestamp"].iloc[icue + 1]))
            ].index.tolist()
            event_df["Timestamp"].loc[idx_events_after_cue] *= scale
        elif icue == (len(event_cues) - 1):
            scale = v_cue / e_cue
            idx_events_after_cue = event_df.loc[
                (event_df["Timestamp"] >= e_cue)
            ].index.tolist()
            event_df["Timestamp"].loc[idx_events_after_cue] *= scale
    new_event_cues = extract_cues_from_events(event_df)

    return event_cues, new_event_cues, voltage_cues


def get_cell_only_activity(input_trace: list, is_cell: list, num_planes: int):
    """
    Returns cell only activity for traces and spikes based on is_cell, 1==cell, 0==not cell in is_cell, for each plane.

    """
    trace_cell_only = []

    for ip in range(num_planes):
        cell_idx = is_cell[ip][:, 0] == 1
        trace_cell_only.append(input_trace[ip][cell_idx, :])

    return trace_cell_only

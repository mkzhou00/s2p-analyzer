import pandas as pd
import numpy as np
import os
import re
import scipy.stats as stats
from scipy.interpolate import interp1d
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.manifold import spectral_embedding   # same code SC uses
from sklearn.cluster._spectral import discretize 


def get_cell_indices(iscell):
    """ Get indices of cells for each plane. """
    return [np.where(plane[:, 0] == 1)[0] for plane in iscell]


def get_cell_coordinates(stat):
    """ Get x, y coordinates as a set of tuples for a given cell. """
    xpix = stat["xpix"]
    ypix = stat["ypix"]
    assert len(xpix) == len(ypix), "xpix and ypix length mismatch!"
    return set(zip(xpix, ypix))


def correct_overlapping_cells_across_planes(stat, iscell, num_planes: int, overlap_threshold=0.8):
    """
    Correct overlapping cells across planes.
    Cells are marked as non-cells in iscell if they have > overlap_threshold overlap.
    
    Args:
        stat: List of dictionaries containing cell coordinates per plane.
        iscell: List of arrays indicating whether each ROI is a cell.
        num_planes: Total number of planes.
        overlap_threshold: Fraction of overlap to be considered the same cell.
        
    Returns:
        overlapping_cells: List of overlapping cell pairs [cell_in_plane_i, cell_in_plane_i+1].
    """
    cellidx = get_cell_indices(iscell)
    overlapping_cells = []

    for ip in range(num_planes - 1):
        for icell in cellidx[ip]:
            ref_coords = get_cell_coordinates(stat[ip][icell])

            for ic in cellidx[ip + 1]:
                target_coords = get_cell_coordinates(stat[ip + 1][ic])

                # Calculate overlap using set intersection
                overlap = ref_coords & target_coords
                overlap_fraction = len(overlap) / len(target_coords)

                # Mark as non-cell if overlap exceeds threshold
                if overlap_fraction >= overlap_threshold:
                    iscell[ip + 1][ic][0] = 0
                    overlapping_cells.append([icell, ic])

    return overlapping_cells


# def correct_overlapping_cells_across_planes(stat, iscell, num_planes:int):
#     cellidx = []
#     for ip in range(len(iscell)):
#         # get the index number of ROIs that is cell (1 in first item of iscell)
#         temp = (iscell[ip] == 1).nonzero()[0]
#         cellidx.append(temp)

#     xy_plane0 = set()
#     overlapping_cells = []
#     for ip in range(num_planes-1):
#         for icell in cellidx[ip]:
#             assert len(stat[ip][icell]["xpix"]) == len(
#                 stat[ip][icell]["ypix"]
#             )
#             xpix = stat[ip][icell]["xpix"]
#             ypix = stat[ip][icell]["ypix"]
#             reference_cell_coordinates = set()
#             for ix, iy in zip(xpix, ypix):
#                 reference_cell_coordinates.add((ix, iy))
#             for ic in cellidx[ip+1]:
#                 assert len(stat[ip+1][ic]["xpix"]) == len(
#                     stat[ip+1][ic]["ypix"]
#                 )
#                 x = stat[ip+1][ic]["xpix"]
#                 y = stat[ip+1][ic]["ypix"]
#                 overlap_ct = 0
#                 for ix, iy in zip(x, y):
#                     if (ix, iy) in reference_cell_coordinates:
#                         overlap_ct += 1
#                 if (overlap_ct / len(x)) >= 0.8:
#                     iscell[ip+1][ic][0] = 0
#                     overlapping_cells.append([icell, ic])

#     return overlapping_cells


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
    # assert len(event_cues) == len(voltage_cues), (
    #     "Cues extracted from events and voltages must match."
    #     f"Event Cues: {len(event_cues)},"
    #     f"voltage cues: {len(voltage_cues)}"
    # )
    return event_cues, voltage_cues


def correct_timestamps(event_df: pd.DataFrame, images, numplanes, imaging_system="INSS", voltages=None):
    """
    Arduino time drifts (assume linear) w.r.t. computer time. Correct timestamps
    collected on Arduino given corresponding computer timestamps.

    Imaging timestamps also drifts, correct based on voltage time stamps

    Args:
        event_df[in/out]: Events w/ timestamps collected on Arduino.
        voltages[in]: Corresponding timestamps collected on computer.
        images[in/out]: image timestamps collected on computer
    """
    if imaging_system == "Bruker":
        ### Voltages (computer received)

        # Extract in-session ts between start (1) and end (0).
        v_in_session = voltages[voltages[" TTL1"] > 3]

        # Get start ts.
        v_session_first_ts = v_in_session.iloc[0][0]

        # Subtract all ts using first timestamp, effectively making event starts at 0.
        v_in_session["Time(ms)"] = v_in_session["Time(ms)"] - v_session_first_ts

        # Do the same thing for Events (on Arduino), setting event start at 0.
        e_session_start = event_df.loc[event_df["Events"] == 1]
        event_df["Timestamp"] = (
            event_df["Timestamp"].to_numpy() - e_session_start["Timestamp"].to_numpy()
        )

        # Now that both data starts at 0, correct the linear drift based on event cues.
        event_cues, voltage_cues = extract_cues(event_df, voltages)
        assert len(event_cues) > 0

        ## Linear scaling just using start and end time points from voltage and events
        # scale = 0
        # end_v = v_in_session["Time(ms)"].iloc[-1]
        # end_e = event_df["Timestamp"].iloc[-1]
        # scale = end_v / end_e
        # event_df["Timestamp"] *= scale

        ## Non linear scaling across time points, correct for each cue
        if len(event_cues) == len(voltage_cues):
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
        
        # Scaling image points
        scale = 0
        end_v = voltages["Time(ms)"].iloc[-1] / 1e3
        if numplanes > 1:
            end_im = max(images[ip][-1] for ip in range(numplanes))
        else:
            end_im = images[-1]
        scale = end_v / end_im
        new_images = []
        for ip in range(len(images)):
            sublist = (np.array(images[ip]) * scale).tolist()
            new_images.append(sublist)    

        return event_df, new_images
    
    else:
        scale = 0
        end_im = images
        end_e = event_df["Timestamp"].iloc[-1] / 1E3
        scale = end_im / end_e
        event_df["Timestamp"] *= scale    
        
        return event_df


def extract_events(event_df: pd.DataFrame):
    # get all events in seconds
    licks = np.array(event_df["Timestamp"][event_df["Events"] == 5] / 1e3)
    CS1 = np.array(event_df["Timestamp"][event_df["Events"] == 15] / 1e3)
    CS2 = np.array(event_df["Timestamp"][event_df["Events"] == 16] / 1e3)
    CS3 = np.array(event_df["Timestamp"][event_df["Events"] == 17] / 1e3)
    sucrose = np.array(
        event_df["Timestamp"][(event_df["Events"] == 10) & (event_df["Reward"] == 0)]
        / 1e3
    )
    milk = np.array(event_df["Timestamp"][
        (event_df["Events"].isin([8, 9])) & (event_df["Reward"] == 0)] / 1e3)
    
    return licks, CS1, CS2, CS3, sucrose, milk

def get_cell_only_activity(F: list, Fneu: list, spks:list, is_cell: list, num_planes: int):
    """
    Returns cell only activity for traces and spikes based on is_cell, 1==cell, 0==not cell in is_cell, for each plane.

    """
    threshold = 0.03  # percentage of F higher than Fneu required to classify as cell
    F_cell = [[] for _ in range(num_planes)]
    Fneu_cell = [[] for _ in range(num_planes)]
    spks_cell = [[] for _ in range(num_planes)]
    passed_idx_by_plane = [[] for _ in range(num_planes)]

    for ip in range(num_planes):
        cell_idx = [index for index, value in enumerate(is_cell[ip]) if value[0] == 1]
        for cell in cell_idx:
            if np.mean(F[ip][cell, :]) > np.mean(
                Fneu[ip][cell, :]
            ) + threshold * np.mean(Fneu[ip][cell, :]):
                F_cell[ip].append(F[ip][cell, :])
                Fneu_cell[ip].append(Fneu[ip][cell, :])
                spks_cell[ip].append(spks[ip][cell,:])
                passed_idx_by_plane[ip].append(cell)

    return F_cell, Fneu_cell, spks_cell, passed_idx_by_plane


def get_corrected_F(F_cell: list, Fneu_cell: list, num_planes: int, coeff: float):
    Fcorr = [[] for _ in range(num_planes)]
    for ip in range(num_planes):
        for ic, (fc, fneu) in enumerate(zip(F_cell[ip], Fneu_cell[ip])):
            # equation for neuro pil F correction
            temp = fc - fneu * coeff
            Fcorr[ip].append(temp)
    return Fcorr


def extract_interest_time_intervals(event_cues, pre_cue_window, post_cue_window):
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
                    cue - pre_cue_window,
                    cue + post_cue_window,
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
                # # find the images condition during each interval
                # if num_planes == 1:
                #     condition_idx = (im_ts >= interval[0]) & (im_ts <= interval[1])
                #     cue_temp = [
                #         i
                #         for i, (ts, condition) in enumerate(zip(im_ts, condition_idx))
                #         if condition
                #     ]
                # else:
                condition_idx = (im_ts[ip] >= interval[0]) & (
                    im_ts[ip] <= interval[1]
                )
                # get the image time points for each cue
                cue_temp = [
                    i
                    for i, (ts, condition) in enumerate(
                        zip(im_ts[0], condition_idx)
                    )
                    if condition
                ]
                # append image time points for each cue under correct CS type and plane
                im_idx_around_cues[ip][cs_type].append(cue_temp)
    im_idx_around_cues = np.array(im_idx_around_cues)
    return im_idx_around_cues


def normalize_signal(Fcorr, num_planes: int, norm_by="median"):
    """
    This function normalizes Fcorrected traces.

    """
    if norm_by == "z_score":
        for ip in range(num_planes):
            mean = np.nanmean(Fcorr[ip], axis=1)
            std = np.std(Fcorr[ip], axis=1)
            Fcorr[ip] = (Fcorr[ip] - mean[:, None]) / std[:, None]
    elif norm_by == "median":
        for ip in range(num_planes):
            median = np.median(Fcorr[ip], axis=1)
            max = np.max(Fcorr[ip], axis=1)
            min = np.min(Fcorr[ip], axis=1)
            Fcorr[ip] = (Fcorr[ip] - median[:, None]) / (max[:, None] - min[:, None])
    elif norm_by == "robust_z_score":
        for ip in range(num_planes):
            median = np.median(Fcorr[ip], axis=1)
            mad = stats.median_absolute_deviation(Fcorr[ip], axis=1)
            Fcorr = 0.6745 * (Fcorr[ip] - median[:, None]) / np.median(mad)
    return Fcorr


def extract_Fave_around_events(
    CS,
    F,
    im_ts,
    num_planes: int,
    pre_cue_window: int,
    post_cue_window: int,
):
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
    Fcorrected_around_cue with the structure of len(CS), number of cells, timepoints

    """
    
    # Extract time around each cue and sorted by CS type, shape is numCS --> len trials
    interest_intervals = extract_interest_time_intervals(
        CS, pre_cue_window, post_cue_window
    )
    # Extract image time points around each cue and sorted by CS type and plane, shape is plane --> numCS --> len trials
    im_idx_around_cue = extract_imaging_ts_around_events(
        CS, im_ts, num_planes, interest_intervals
    )

    # F_ave_around_cues = [[] for _ in range(len(CS))]
    F_ave_around_cues_baseline_subtract = [[] for _ in range(len(CS))]

    framenumber = len(
        F[0][0][im_idx_around_cue[0][0][1]]
    )  # reference frame number equals the first cell's second trial from the first plane
    framespersecond = framenumber // (pre_cue_window + post_cue_window)

    for cue_type, cs in enumerate(CS):  # cue_type = 0,1,2 (CS1, CS2, CS3)
        for ip in range(num_planes):
            cue_ts = im_idx_around_cue[ip][
                cue_type
            ]  # image indexes for all trials in this cue type, holds same for all cells within the plane (trial number x framenumber)
            for cell in range(len(F[ip])):
                cell_F = []
                for trial in range(len(cs)):
                    F_temp = F[ip][cell][
                        cue_ts[trial]
                    ]  # F for cell in the plane, of this trial in this cue type (framenumber x )
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
                # baseline = np.nanmean(cellave[0 : pre_cue_window * framespersecond])
                # baselinesubtract = list(cellave - baseline)
                F_ave_around_cues_baseline_subtract[cue_type].append(cellave)
                # F_ave_around_cues[cue_type].append(cellave)
    F_ave_around_cues_baseline_subtract = np.array(F_ave_around_cues_baseline_subtract)
    return F_ave_around_cues_baseline_subtract


def extract_F_around_events(
    CS,
    F,
    im_ts,
    num_planes: int,
    pre_cue_window: int,
    post_cue_window: int,
    binsize=None,
    framerate=5
):
    """
    This function extract F traces around cue events, with optional binning in milliseconds for decoding.

    Args:
        CS: list of CS trials grouped by type
        F: calcium trace data [plane][cell][frame]
        im_ts: image timestamps
        num_planes: number of imaging planes
        pre_cue_window: seconds before cue onset
        post_cue_window: seconds after cue onset
        binsize: bin size in milliseconds (set to None to disable binning)
        framerate: frame rate in Hz

    Returns:
        F_trial: array with shape 
                 [n_CS_types][n_trials][n_cells][n_timepoints]
    """
    windowsize = pre_cue_window + post_cue_window
    total_frames = int(windowsize * framerate)    
    
    # Binning to get average F for each bin, this is for decoding purposes mostly
    do_binning = binsize is not None and binsize > 0
    binframes = int((binsize / 1000) * framerate) if do_binning else 1
    if do_binning and binframes < 1:
        raise ValueError("binsize too small for given framerate — results in < 1 frame/bin.")

    # Extract time around each cue and sorted by CS type, shape is numCS --> len trials
    interest_intervals = extract_interest_time_intervals(
        CS, pre_cue_window, post_cue_window
    )
    # Extract image time points around each cue and sorted by CS type and plane, shape is plane --> numCS --> len trials
    im_idx_around_cue = extract_imaging_ts_around_events(
        CS, im_ts, num_planes, interest_intervals
    )

    # Create a list for each cs type and each trial, using None to hold the place. Shape is nCS --> ntrial within each CS
    # F_trial = [[] for _ in CS]
    # mintrial_for_decoding = min(len(cs) for cs in CS)
    # for i in range(len(CS)):
    #     F_trial[i] = [[] for _ in range(mintrial_for_decoding)]
        
    F_trial = [[[] for _ in range(len(cs))] for cs in CS]
                
    for cue_type, cs in enumerate(CS):  # cue_type = 0,1,2 (CS1, CS2, CS3)
        for ip in range(num_planes):
            cue_ts = im_idx_around_cue[ip][
                cue_type
            ]  # image indexes for all trials in this cue type, holds same for all cells within the plane (trial number x framenumber)
            for cell in range(len(F[ip])):
                # cell_F = []
                for trial in range(len(cs)):
                    F_temp = F[ip][cell][
                        cue_ts[trial]
                    ]  # F for cell in the plane, of this trial in this cue type (framenumber x )
                    
                    if do_binning:
                        F_temp_binned = []
                        for ibin in range(0, total_frames, binframes):
                            F_slice = F_temp[ibin:(ibin + binframes)]
                            if F_slice.size == 0 or np.all(np.isnan(F_slice)):
                                F_temp_binned.append(0.0)
                            else:
                                F_temp_binned.append(np.nanmean(F_slice))
                        F_trial[cue_type][trial].append(F_temp_binned)
                    else:
                        F_trial[cue_type][trial].append(F_temp)
            
    # This is padding the function incase the timepoint of certain neuron is not the same with the target time points
    max_len = total_frames
    for cs in range(len(F_trial)):
        for trial in range(len(F_trial[cs])):
            for cell in range(len(F_trial[cs][trial])):
                series = F_trial[cs][trial][cell]
                if len(series) < max_len:
                    padded = np.pad(series, (0, max_len - len(series)), constant_values=0)
                    F_trial[cs][trial][cell] = padded
                elif len(series) > max_len:
                    F_trial[cs][trial][cell] = series[:max_len]
                
    return F_trial


def reorder_clusters(populationdata, pre_window_size, rawlabels):
    uniquelabels = list(set(rawlabels))
    responses = np.nan * np.ones((len(uniquelabels),))
    for l, label in enumerate(uniquelabels):
        responses[l] = np.mean(
            populationdata[rawlabels == label, pre_window_size : 2 * pre_window_size]
        )
    temp = np.argsort(responses).astype(int)[::-1]
    temp = np.array([np.where(temp == a)[0][0] for a in uniquelabels])
    outputlabels = np.array(
        [temp[a] for a in list(np.digitize(rawlabels, uniquelabels) - 1)]
    )
    return outputlabels


def resample_data(F, im_ts, current_rate, target_rate, binning_tolerance=0.2):
   
    F_resampled = []
    im_ts_resampled = []
    
    for ip, F_plane in enumerate(F):
        n_cells, n_timepoints = np.array(F_plane).shape
        t_orig = np.array(im_ts[ip])
        if len(t_orig) != n_timepoints:
            raise ValueError(f"Plane {ip}: length of im_ts does not match number of timepoints")
  
        ratio = current_rate / target_rate
        use_binning = np.isclose(ratio, round(ratio), atol=binning_tolerance)

        # If the current rate can be evenly divided by the target framerate or that its close to the integer with margin of 0.2, use binning
        # to average the values to the first point 
        # if np.remainder(current_rate, target_rate) == 0:
        if (target_rate < current_rate) and np.isclose(ratio, round(ratio), atol=binning_tolerance):    
            bin_size = int(round(ratio))
            n_bins = n_timepoints // bin_size
            if n_bins < 1:
                use_binning = False
            else:
                t_trunc = t_orig[ip][:n_bins * bin_size]
                t_new = t_trunc[::bin_size]
                F_down = np.zeros((n_cells, n_bins))
            
                for i in range(n_cells):
                    f_trunc = F_plane[i][:n_bins * bin_size]
                    binned = f_trunc.reshape(n_bins, bin_size)
                    f_down = np.nanmean(binned, axis=1)           # shape: (n_bins,)
                    F_down[i] = np.round(f_down, 5)
                
                im_ts_resampled.append(t_new)
                F_resampled.append(F_down)
                continue # skip to next plane
         
            # bin_size = int(np.divide(current_rate, target_rate))
            
            # # only include the first time point in each bin
            # t_orig = im_ts[ip]
            # new_n_timepoints = len(t_orig) // bin_size
            # t_new = t_orig[::bin_size][:new_n_timepoints]     
                   
            # F_down = np.zeros((n_cells, new_n_timepoints))
            
            # for i in range(n_cells):
            #     # Reshape into bins and average
            #     binned = F_plane[i][:new_n_timepoints * bin_size].reshape(new_n_timepoints, bin_size)
            #     f_down = np.nanmean(binned, axis=1)
            #     F_down[i] = np.round(f_down, 5)
        
        # if the current rate is not divided evenly or close to integer divide, interpolate time points to fit the target framerate
        # else:
        #     t_orig = im_ts[ip]
        #     duration = t_orig[-1] - t_orig[0]            
        #     new_n_timepoints = int(duration * target_rate)
        #     t_new = np.linspace(t_orig[0], t_orig[-1], new_n_timepoints)
            
        #     F_down = np.zeros((n_cells, new_n_timepoints))

        #     for i in range(n_cells):
        #         f_interp = interp1d(t_orig, F_plane[i], kind='linear')
        #         f_interped = f_interp(t_new)
        #         F_down[i] = np.round(f_interped, 5)
        
        # im_ts_resampled.append(t_new)
        # F_resampled.append(F_down)
        duration = t_orig[-1] - t_orig[0]        
        new_n_timepoints = int(np.round(duration * target_rate))
        # if new_n_timepoints < 2:
        #     new_n_timepoints = 2
        t_new = np.linspace(t_orig[0], t_orig[-1], new_n_timepoints)
        F_down = np.zeros((n_cells, new_n_timepoints))

        for i in range(n_cells):
            # interp1d fails if there are NaNs in the input trace. A minimal strategy: do a simple
            # linear fill of NaNs before interpolation. If all NaNs, keep them.
            trace = F_plane[i].astype(float)
            if np.any(np.isnan(trace)):
                valid = ~np.isnan(trace)
                # If NaNs only at edges, fill with nearest valid value by specifying fill_value
                f_fill = interp1d(t_orig[valid], trace[valid], kind='linear',
                                  bounds_error=False,
                                  fill_value=(trace[valid][0], trace[valid][-1]))
                trace_filled = f_fill(t_orig)
            else:
                trace_filled = trace

            f_interp = interp1d(t_orig, trace_filled, kind='linear',
                                bounds_error=False,
                                fill_value=(trace_filled[0], trace_filled[-1]))
            f_interped = f_interp(t_new)
            F_down[i] = np.round(f_interped, 5)

        im_ts_resampled.append(t_new)
        F_resampled.append(F_down)

    return F_resampled, im_ts_resampled


def filter_trials_by_minITI(cs_events, min_ITI):
    filtered_CS = []
    # Filter all cues with ITI longer than post window here
    for ics, cs in enumerate(cs_events):
        itis = np.diff(cs)
        keep_mask = np.ones(len(cs), dtype=bool)
        # First trial is always kept (no ITI before it)
        keep_mask[1:] = (itis >= min_ITI)
        filtered_CS.append(cs[keep_mask])
        
    return filtered_CS


# def extract_patterns_for_decoding(animal_data, time_slice):
#     """
#     animal_data: np.array of shape (3, n_trials, n_cells, nframes)
#     time_slice: slice object (e.g., slice(0, 5) for cue period)
    
#     Returns:
#         X: (3 * n_trials, n_cells * len(time_slice))
#         y: (3 * n_trials,)
#     """
#     n_CS, n_trials, n_cells, _ = animal_data.shape
#     period_data = animal_data[..., time_slice]  # shape: (3, n_trials, n_cells, n_timepoints)
#     reshaped = period_data.reshape(n_CS * n_trials, n_cells * period_data.shape[-1])
#     labels = np.repeat(np.arange(n_CS), n_trials)  # 0, 1, 2 for CS1+, CS2+, CS3−
#     return reshaped, labels


def extract_patterns_for_decoding(animal_data, time_slice):
    """
    animal_data: list of 3 arrays, each shape (n_trials_cs, n_cells, n_frames)
    time_slice: slice object (e.g., slice(0, 5) for cue period)
    
    Returns:
        X: (total_trials, n_cells * len(time_slice))
        y: (total_trials,)
    """
    X_list = []
    y_list = []
    for cs_type, cs_trials in enumerate(animal_data):
        # cs_trials: shape (n_trials_cs, n_cells, n_frames)
        period_data = cs_trials[..., time_slice]  # (n_trials_cs, n_cells, n_timepoints)
        reshaped = period_data.reshape(period_data.shape[0], -1)  # (n_trials_cs, n_cells * n_timepoints)
        X_list.append(reshaped)
        y_list.append(np.full(period_data.shape[0], cs_type))
    X = np.concatenate(X_list, axis=0)  # (total_trials, n_cells * n_timepoints)
    y = np.concatenate(y_list, axis=0)  # (total_trials,)
    return X, y


def get_animal_decoding_dict(
    animal_list,
    patterns,
    labels,
    trial_types,
    total_bins_length,
    decode_by_cluster=False,
    cluster_labels=None,
    selected_clusters=None,
    animal_id=None,
):
    """
    Prepares a dictionary of decoded animal data for decoding or PSTH plotting.

    Returns:
        decoded_animals (dict): keys are animal IDs, values contain data, local cell indices, and metadata.
    """
    cue_labels = [0, 1, 2]
    
    if decode_by_cluster:
        animal_cell_starts = {}
        start_idx = 0
        for a in animal_list:
            n_cells = np.sum(np.char.find(animal_id.astype(str), a) >= 0)
            animal_cell_starts[a] = start_idx
            start_idx += n_cells
        
    decoded_animals = {}

    for animal in animal_list:
        data = patterns[animal]  # list of arrays, each [n_trials_for_cue, n_cells * total_bins_length]
        n_trial_per_cue = [(labels[animal] == cue).sum() for cue in cue_labels]
        # Check that each array is shaped correctly
        # for d in data:
        #     assert d.shape[1] % total_bins_length == 0, \
                # f"Unexpected number of features in {animal}: {d.shape[1]}"
        total_cells = data[0].shape[0] / total_bins_length # assumes all cues have same cell count

        if decode_by_cluster:
            global_mask = (
                (np.char.find(animal_id.astype(str), animal) >= 0) &
                np.isin(cluster_labels, selected_clusters)
            )
            global_indices = np.where(global_mask)[0]
            local_start = animal_cell_starts[animal]
            local_indices = global_indices - local_start
            
            invalid_mask = local_indices >= total_cells
            if np.any(invalid_mask):
                print(f"[{animal}] Warning: {np.sum(invalid_mask)} cluster-assigned cells exceed available cells ({total_cells}).")
                local_indices = local_indices[~invalid_mask]
            
            if len(local_indices) == 0:
                print(f"[{animal}] No cells in selected clusters {selected_clusters}. Skipping.")
                continue
        else:
            local_indices = np.arange(total_cells)

        decoded_animals[animal] = {
            "data": data,  # list of arrays per cue
            "local_cell_indices": local_indices,
            "n_trial_per_cue": n_trial_per_cue,  # list, one entry per cue type
            "n_cells": len(local_indices),
        }

    return decoded_animals


def do_time_resolved_decoding(
    decoded_animals,
    animal_list,
    decoding_time_window,
    decoding_pair,
    total_bins_length,
    subtrials='all',
    niteration=5,
    clf=LinearSVC(),
    clf_chance=LinearSVC(),   
    subsampling=np.nan,
    seed=42,
    testing_pair=None
):
    """
    Perform time-resolved decoding with leave-one-trial-out cross-validation.

    Args:
        decoded_animals (dict): Output from `get_decoded_animals`, per animal.
        animal_list (list): List of animals to decode.
        decoding_time_window (array): Time bins to decode at.
        decoding_pair (tuple): Trial types to decode (e.g., ("CS1", "CS3")).
        total_bins_length (int): Number of bins per cell.
        niteration (int): Number of iterations (for subsampling).
        subsampling (float or np.nan): Proportion of cells to subsample.
        clf (sklearn classifier): Main classifier.
        clf_chance (sklearn classifier): For chance decoding.
        seed (int): Random seed.

    Returns:
        accuracy (np.ndarray): Shape (n_animals, n_timepoints).
        accuracy_chance (np.ndarray): Same shape, for shuffled labels.
    """
    trial_types = ["CS1", "CS2", "CS3"]
    accuracy = [[] for _ in animal_list]
    accuracy_chance = [[] for _ in animal_list]

    for t in decoding_time_window:
        for ia, animal in enumerate(animal_list):

            d = decoded_animals[animal]
            data = d["data"]
            local_cell_indices = d["local_cell_indices"]
            n_cells = d["n_cells"]
            total_trial_per_cue = d["n_trial_per_cue"]
            
            time_indices = local_cell_indices * total_bins_length + t
            starts = np.cumsum([0] + list(total_trial_per_cue[:-1]))
            ends = np.cumsum(list(total_trial_per_cue))
            cue_map = {
                cue: data[starts[i]:ends[i], time_indices]
                for i, cue in enumerate(trial_types)
            }

            train_a = cue_map[decoding_pair[0]]
            train_b_raw = cue_map[decoding_pair[1]]

            performance = []
            performance_chance = []

            for iiter in range(niteration):
                # Subsampling for CS3
                # for CS3, randomly sample n trials for this round
                train_b = train_b_raw
                if decoding_pair[1] == "CS3" and train_b_raw.shape[0] > train_a.shape[0]:
                    np.random.seed(seed + iiter)
                    selected_idx = np.random.choice(train_b_raw.shape[0], train_a.shape[0], replace=False)
                    train_b = train_b_raw[selected_idx]
                    
                # Apply subtrials after sampling
                if subtrials == 'first10':
                    train_a = train_a[:10]
                    train_b = train_b[:10]
                elif subtrials == 'last10':
                    train_a = train_a[-10:]
                    train_b = train_b[-10:]
                elif isinstance(subtrials, (list, np.ndarray)):
                    train_a = train_a[subtrials]
                    train_b = train_b[subtrials]
                
                n_trial_per_cue = train_a.shape[0]

                # Cross validation
                performance_temp = []
                performance_chance_temp = []      
                if np.isnan(subsampling):
                    cell_idx = np.arange(n_cells)
                else:
                    n_sub = int(n_cells * subsampling)
                    cell_idx = np.random.choice(n_cells, n_sub, replace=False)

                for itrial in range(n_trial_per_cue):
                    cs_a_train = np.delete(train_a, itrial, axis=0)[:, cell_idx]
                    cs_b_train = np.delete(train_b, itrial, axis=0)[:, cell_idx]

                    traindata = np.vstack((cs_a_train, cs_b_train))
                    trainlabel = np.array([0] * (n_trial_per_cue - 1) + [1] * (n_trial_per_cue - 1))

                    if traindata.shape[0] != len(trainlabel):
                        raise ValueError("Mismatch between traindata rows and trainlabel length")
                    
                    testdata = np.vstack((train_a[itrial, cell_idx], train_b[itrial, cell_idx]))

                    clf.fit(traindata, trainlabel)
                    testlabel = clf.predict(testdata)
                    performance_temp.append(testlabel == [0, 1])

                    np.random.seed(seed+iiter)
                    shufflelabel = np.random.permutation(trainlabel)
                    clf_chance.fit(traindata, shufflelabel)
                    testlabel = clf_chance.predict(testdata)
                    performance_chance_temp.append(testlabel == [0, 1])

                performance.append(np.mean(np.concatenate(performance_temp)))
                performance_chance.append(np.mean(np.concatenate(performance_chance_temp)))

            accuracy[ia].append(np.mean(performance))
            accuracy_chance[ia].append(np.mean(performance_chance))
            # scores, chance_scores = decode_within(sliced_patterns, labels, n_loops=5)  
            # accuracy.append(np.mean(scores))
            # chance_results.append(np.mean(chance_scores))

    return np.array(accuracy), np.array(accuracy_chance)


def do_time_window_decoding(
    decoded_animals,
    animal_list,
    time_window,  # e.g. [7, 8, 9]
    decoding_pair,
    total_bins_length,
    subtrials='all',
    niteration=5,
    clf=LinearSVC(),
    clf_chance=LinearSVC(),
    subsampling=np.nan,
    seed=42,
    testing_pair=None
):
    """
    Perform time-window decoding with leave-one-trial-out cross-validation.

    Args:
        decoded_animals (dict): Output from `get_decoded_animals`, per animal.
        animal_list (list): List of animals to decode.
        time_window (list): Time bin indices to use (e.g., [7, 8, 9]).
        decoding_pair (tuple): Training CS labels (e.g., ("CS1", "CS3")).
        total_bins_length (int): Number of bins per cell.
        subtrials (str/list): Which trials to use for training.
        niteration (int): Iterations for subsampling.
        clf (sklearn classifier): Main classifier.
        clf_chance (sklearn classifier): Chance classifier.
        subsampling (float or np.nan): Proportion of cells to sample.
        seed (int): Random seed.
        testing_pair (tuple or None): If different from training pair.

    Returns:
        accuracy (np.ndarray): Shape (n_animals,)
        accuracy_chance (np.ndarray): Shape (n_animals,)
    """
    accuracy = []
    accuracy_chance = []

    for ia, animal in enumerate(animal_list):
        d = decoded_animals[animal]
        data = d["data"]
        local_cell_indices = d["local_cell_indices"]
        n_cells = d["n_cells"]
        n_trial_per_cue = d["n_trial_per_cue"]

        # Build time indices for all cells across time_window
        time_indices = np.concatenate([
            local_cell_indices * total_bins_length + t for t in time_window
        ])

        cue_map = {
            "CS1": data[0:n_trial_per_cue, :][:, time_indices],
            "CS2": data[n_trial_per_cue: 2 * n_trial_per_cue, :][:, time_indices],
            "CS3": data[2 * n_trial_per_cue: 3 * n_trial_per_cue, :][:, time_indices],
        }

        # Assign training and testing cues
        train_a = cue_map[decoding_pair[0]]
        train_b = cue_map[decoding_pair[1]]

        if testing_pair is None:
            test_a = train_a
            test_b = train_b
        else:
            test_a = cue_map[testing_pair[0]]
            test_b = cue_map[testing_pair[1]]

        # Subset of trials to decode
        if subtrials == 'first10':
            train_a = train_a[:10]
            train_b = train_b[:10]
        elif subtrials == 'last10':
            train_a = train_a[-10:]
            train_b = train_b[-10:]
        elif isinstance(subtrials, (list, np.ndarray)):
            train_a = train_a[subtrials]
            train_b = train_b[subtrials]
        elif subtrials == 'all' or subtrials is None:
            pass  # use all trials
        else:
            raise ValueError(f"Invalid subtrials value: {subtrials}")

        n_train_trials = train_a.shape[0]
        n_test_trials = min(test_a.shape[0], test_b.shape[0])

        performance = []
        performance_chance = []

        for iiter in range(niteration):
            if np.isnan(subsampling):
                cell_idx = np.arange(n_cells)
            else:
                n_sub = int(n_cells * subsampling)
                cell_idx = np.random.choice(n_cells, n_sub, replace=False)

            # Adjust time indices to match selected cells
            selected_time_indices = np.concatenate([
                local_cell_indices[cell_idx] * total_bins_length + t for t in time_window
            ])

            n_time_bins = len(time_window)
            feature_idx = np.concatenate([cell_idx * n_time_bins + i for i in range(n_time_bins)])

            train_a_feat = train_a[:, feature_idx]
            train_b_feat = train_b[:, feature_idx]
            test_a_feat = test_a[:, feature_idx]
            test_b_feat = test_b[:, feature_idx]

            performance_temp = []
            performance_chance_temp = []

            np.random.seed(seed + iiter)

            for itrial in range(n_test_trials):
                if itrial < n_train_trials:
                    cs_a_train = np.delete(train_a_feat, itrial, axis=0)
                    cs_b_train = np.delete(train_b_feat, itrial, axis=0)
                else:
                    cs_a_train = train_a_feat
                    cs_b_train = train_b_feat

                traindata = np.vstack((cs_a_train, cs_b_train))
                trainlabel = np.array([0] * cs_a_train.shape[0] + [1] * cs_b_train.shape[0])

                testdata = np.vstack((test_a_feat[itrial, :], test_b_feat[itrial, :]))
                testlabel_true = np.array([0, 1])

                clf.fit(traindata, trainlabel)
                testlabel = clf.predict(testdata)
                performance_temp.append(testlabel == testlabel_true)

                shufflelabel = np.random.permutation(trainlabel)
                clf_chance.fit(traindata, shufflelabel)
                testlabel_chance = clf_chance.predict(testdata)
                performance_chance_temp.append(testlabel_chance == testlabel_true)

            performance.append(np.mean(np.concatenate(performance_temp)))
            performance_chance.append(np.mean(np.concatenate(performance_chance_temp)))

        accuracy.append(np.mean(performance))
        accuracy_chance.append(np.mean(performance_chance))

    return np.array(accuracy), np.array(accuracy_chance)


def build_knn(X, k):
    """Return sparse k-NN connectivity graph (CSR)."""
    return kneighbors_graph(
        X, k,
        mode="connectivity",
        include_self=True,    # matches SpectralClustering default
        n_jobs=-1
    )
    

def get_initial_cluster_labels(X, clustering_model, n_clusters, n_neighbors=None, G=None):
    """
    Run clustering and return labels based on the specified model.

    Args:
        X (ndarray): Transformed data (n_samples, n_features).
        clustering_model (str): Model type string.
        n_clusters (int): Number of clusters to fit.
        n_neighbors (int or None): Number of neighbors (for Spectral).
        G (ndarray or None): Precomputed affinity matrix (for discretize).

    Returns:
        labels (ndarray): Cluster labels for each sample.
    """
    if clustering_model == "SC_discretize":
        if G is None:
            raise ValueError("Affinity matrix G must be provided for discretize spectral clustering.")
        embed = spectral_embedding(
            G,
            n_components=n_clusters,
            eigen_solver="arpack",
            drop_first=False
        )
        labels = discretize(embed, random_state=None)

    elif clustering_model == "SC_kmeans":
        if n_neighbors is None:
            raise ValueError("n_neighbors must be provided for SpectralClustering.")
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels='kmeans'
        )
        model.fit(X)
        labels = model.labels_

    elif clustering_model == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)

    elif clustering_model == "AgglomerativeClustering":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='l1',
            linkage='average'
        )
        labels = model.fit_predict(X)

    else:
        raise ValueError(f"Unsupported clustering model: {clustering_model}")

    return labels



def get_filtered_rois_per_animal_plane(animal_list, day_list, data_dir, oldlabels, target_day_list, skip_if_missing_plane_day=None):
    """
    Returns
    -------
    filtered_rois_per_animal_plane : dict
        Format:
            {
                animal1: {plane0: [roi_idx0, ...], plane1: [...] ...},
                animal2: {plane0: [...], ...},
                ...
            }
        Each list contains cell indices (0-based within each plane) ON DAY 1, which also have matches on the trained day
    """
    filtered_rois_per_animal_plane = {}
    newlabels_flag = {}
    
    if target_day_list is None:
        per_animal_req_day = None # use all days per animal
    else:
        per_animal_req_day = target_day_list
    
    for a, (day, animal) in enumerate(zip(day_list, animal_list)):
        top_animal_dir = os.path.join(data_dir, animal)
        main_dir = os.path.join(top_animal_dir, f"d{day}")
        file_dir = os.path.join(main_dir, "files")
        
        print(animal)
        # Load cell index for this trained day
        cells_idx = np.load(os.path.join(file_dir, "cell_idx.npy"), allow_pickle=True)
        num_planes = len(cells_idx)
        
        
        # Plane offsets for global indexing
        plane_offsets_for_global_idx = np.cumsum([0] + [len(x) for x in cells_idx[:-1]])
        filtered_rois_per_animal_plane[animal] = {}
        newlabels_flag[animal] = {}
        
        trained_day_col = str(day-1)
        
        for ip in range(num_planes):
            # Load ROI table for each animal and the day columns 
            ROI_table_across_sessions = pd.read_csv(os.path.join(top_animal_dir, f"ROI_table_full_plane{ip}.csv"))
            all_day_cols = [col for col in ROI_table_across_sessions.columns if re.match(r'^\d+$', str(col).strip())]

            # Find the rows where the trained day column matches the current plane's cell indices for all the target sessions
            matched_rows_on_trained_day = ROI_table_across_sessions[ROI_table_across_sessions[trained_day_col].isin(cells_idx[ip])]
            matched_rows_on_trained_day = matched_rows_on_trained_day.sort_values(by=trained_day_col)
            newlabels_flag[animal][ip] = np.full(len(matched_rows_on_trained_day), fill_value=-1)  # Initialize with -1
            
            # Skip the plane if data is missing for some days
            if skip_if_missing_plane_day and animal in skip_if_missing_plane_day:
                plane_to_skip = skip_if_missing_plane_day[animal]
                if plane_to_skip == ip:
                    print(f"Skipping animal {animal}, plane {ip}")
                    filtered_rois_per_animal_plane[animal][ip] = []
                    continue
                
            # correct for 0-based indexing
            if per_animal_req_day[a] is None:
                required_day_cols = all_day_cols
            else:
                required_day_cols = [str(d-1) for d in per_animal_req_day[a] if str(d) in all_day_cols]
            
            
            # Get the ROIs that also have matches on target day thats not -1 (both int or str)
            all_valid_mask = (matched_rows_on_trained_day[required_day_cols].astype(str) != '-1').all(axis=1)
            index_filtered_ROIs = np.flatnonzero(all_valid_mask.values).tolist()
            filtered_ROIs = matched_rows_on_trained_day[all_valid_mask].copy()
            
            # Get the remaining cell indices on trained day based on the target days
            remaining_cells = filtered_ROIs[trained_day_col].astype(int).values.tolist()
            
            # Get the new labels for these cells and sort by day 1 index
            global_indices = [plane_offsets_for_global_idx[ip] + idx for idx in remaining_cells]
            assert(len(global_indices) == len(filtered_ROIs)), "Length mismatch!"
            
            labels = np.asarray(oldlabels)[global_indices].tolist()
            filtered_ROIs['labels'] = labels

            # filtered_ROIs = filtered_ROIs.sort_values(by=trained_day_col)
            
            # # reorder labels accordingly and save as new labels
            # reordered_labels = filtered_ROIs['labels'].values.tolist()
            # newlabels_per_animal_plane.extend(labels)
            
            filtered_rois_per_animal_plane[animal][ip] = filtered_ROIs
            for row, label in zip(index_filtered_ROIs, labels):
                newlabels_flag[animal][ip][row] = label
            
            assert (len(filtered_rois_per_animal_plane[animal][ip]) == len(labels)), "Length mismatch!"
            # Now filtered_rois_per_animal_plane[animal][ip] is the list of cell indices within plane ip for animal, present on day 1 and in ROI table

    return filtered_rois_per_animal_plane, newlabels_flag


def load_population_data_filtered(
    animal_list, 
    day_list, 
    data_dir, 
    result_dir, 
    target_frames, 
    pre_cue_window, 
    framerate, 
    filtered_rois_per_animal_plane,   # dict: animal -> plane -> day -> ROI list
    subtrials=None
):

    populationdata_list = []
    animal_id = []
    foundcells_idx_full = []
    foundcells_flags_full = []
    new_labels_on_test_day = []  
                   
    for animal, day in zip(animal_list, day_list):

        print(f"\n=== Loading {animal} (day {day}) ===")
        file_dir = os.path.join(data_dir, animal, f"d{day}", "files")

        # --- Load raw data ---
        rawdata = np.load(os.path.join(file_dir, "F_around_cue_raw.npy"), allow_pickle=True)
        cells_idx = np.load(os.path.join(file_dir, "cell_idx.npy"), allow_pickle=True)

        # Concatenate mapping: plane0 cells, plane1 cells, ...
        plane_offsets = np.cumsum([0] + [len(x) for x in cells_idx[:-1]])
        cell_list_full = np.concatenate([np.asarray(x) for x in cells_idx])

        total_cells = len(cell_list_full)
        print(f"Total cells: {total_cells}")

        foundcell_flags_per_animal = np.zeros(total_cells, dtype=int)

        # --- Build trial subsets ---
        n_trials = [len(rawdata[i]) for i in range(rawdata.shape[0])]
        min_trials = min(n_trials)
        rng = np.random.default_rng()

        idxs_per_cue = []
        if subtrials in (None, "all"):
            K = min_trials
            for n in n_trials:
                if n == K:
                    idxs_per_cue.append(np.arange(n))
                else:
                    idxs = np.sort(rng.choice(n, size=K, replace=False))
                    idxs_per_cue.append(idxs)

        elif subtrials == "first10":
            K = min(10, min_trials)
            for n in n_trials:
                idxs_per_cue.append(np.arange(min(n, K)))

        elif subtrials == "last10":
            K = min(10, min_trials)
            for n in n_trials:
                start = max(0, n - K)
                idxs_per_cue.append(np.arange(start, n))

        # Subset & baseline
        subset_raw = [np.take(a, idx, axis=0) for a, idx in zip(rawdata, idxs_per_cue)]
        subset_ave = np.mean(subset_raw, axis=1)  # (trial_types, ncells, nframes)

        baseline = np.mean(subset_ave[:, :, :int(pre_cue_window * framerate)], axis=2, keepdims=True)
        subset_ave = subset_ave - baseline

        # Rearrange to (ncells, trial_types, nframes)
        subset_by_cell = subset_ave.transpose(1, 0, 2)

        # --- Process each plane ---
        roi_table_for_animal = filtered_rois_per_animal_plane[animal]

        for ip, plane_roi_dict in roi_table_for_animal.items():

            # skip nonexistent planes
            if ip >= len(cells_idx):
                continue

            # extract ROIs for test day
            if str(day-1) not in plane_roi_dict:
                continue

            ROI_on_test_day = plane_roi_dict[str(day-1)].astype(int)
            labels_on_test_day = plane_roi_dict['labels'].astype(int)
            if len(ROI_on_test_day) == 0:
                continue

            # local indices for this plane
            cells_idx_this_plane = np.array(cells_idx[ip])

            found_cells_local = []
            found_flags = []

            # match input ROIs to actual existing cell IDs
            for cellnum in ROI_on_test_day:
                match = np.where(cells_idx_this_plane == cellnum)[0]
                found_flags.append(1 if match.size > 0 else 0)
                if match.size > 0:
                    found_cells_local.append(match[0])

            found_cells_local = np.array(found_cells_local)
            if found_cells_local.size == 0:
                continue

            # Map local → global using plane offsets
            global_indices = plane_offsets[ip] + found_cells_local
            foundcell_flags_per_animal[global_indices] = 1
            
            # collect cluster labels if the cell is found on this day
            new_labels_on_test_day.extend(labels_on_test_day[np.array(found_flags) == 1])
            
            # Extract data: (n_cells, trial_types, nframes)
            tempdata = subset_by_cell[global_indices]

            # Flatten trialtypes × frames
            n_cells = tempdata.shape[0]
            tempdata = tempdata.reshape(n_cells, -1)

            # Pad/crop to target_frames
            if tempdata.shape[1] < target_frames:
                pad = target_frames - tempdata.shape[1]
                tempdata = np.pad(tempdata, ((0, 0), (0, pad)))
            else:
                tempdata = tempdata[:, :target_frames]

            populationdata_list.append(tempdata)
            animal_id.extend([animal] * len(global_indices))
            foundcells_idx_full.extend(global_indices.tolist())
            foundcells_flags_full.extend(found_flags)

        # append flags for this animal
        foundcells_flags_full.extend(foundcell_flags_per_animal.tolist())

    # --- Final assembly ---
    if populationdata_list:
        populationdata = np.vstack(populationdata_list)
        animal_id = np.array(animal_id)
    else:
        populationdata = np.zeros((0, target_frames))
        animal_id = np.array([])
    assert(len(populationdata) == len(new_labels_on_test_day)), "Length mismatch between population data and new labels!"

    return populationdata, animal_id, foundcells_idx_full, foundcells_flags_full, new_labels_on_test_day

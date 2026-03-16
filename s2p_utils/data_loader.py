import logging
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import xml.etree.ElementTree as ET
import glob
from ScanImageTiffReader import ScanImageTiffReader as sctiffreader
import re


logger = logging.getLogger(__name__)


def get_file_with_type(type: str, dir: str) -> str:
    """
    Returns the first file found in the folder matching the file type

    Args:
        type: File type.
        dir: Directory to search in.
    """
    for file in os.listdir(dir):
        if file.endswith(type):
            return dir + "/" + file
    return None


class DataLoader:
    def __init__(self, data_dir: str, num_planes: int, num_flyback:int, imaging_system: str) -> None:
        """
        Data set inclduing Suite2p, behavioral, and image ts. All the files
        are lazy loaded upon request.

        Args:
            data_dir: Data directory containing all the required files.
        """
        self.data_dir = data_dir
        self.s2p_dir = os.path.join(data_dir, "suite2p")
        self.file_dir = os.path.join(data_dir, "files")
        self.plane_subfolder = "plane"
        self.num_plane = num_planes
        self.num_flyback = num_flyback
        self.imaging_system = imaging_system

        # Suite2p output.
        self.F_all = []

        # Behavioral data. (mat)
        self.behave = None
        self.event_df = None

        # Image timestamps. (xml)
        self.im_ts = []

        # Session start and end timestamps. (csv)
        self.voltages = None

    # Get suite2p generated files from s2p folders for each plane.
    def _load_F_all(self):
        if len(self.F_all) == 0:
            for ip in range(self.num_plane):
                plane_dir = os.path.join(self.s2p_dir, self.plane_subfolder + str(ip))

                # Check and load individual .npy files
                plane_data = {}
                for key in ['F', 'Fneu', 'spks', 'stat', 'ops', 'iscell']:
                    file_path = os.path.join(plane_dir, f"{key}.npy")
                    if os.path.exists(file_path):
                        plane_data[key] = np.load(file_path, allow_pickle=True)
                    else:
                        print(f"Warning: {file_path} not found!")

                self.F_all.append(plane_data)
        return self.F_all

    def get_F(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        F = [plane['F'] for plane in self.F_all if 'F' in plane]
        return F

    def get_Fneu(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        Fneu = [plane['Fneu'] for plane in self.F_all if 'Fneu' in plane]
        return Fneu

    def get_spks(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        spks = [plane['spks'] for plane in self.F_all if 'spks' in plane]
        return spks

    def get_stat(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        stat = [plane['stat'] for plane in self.F_all if 'stat' in plane]
        return stat

    def get_ops(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        ops = [plane['ops'] for plane in self.F_all if 'ops' in plane]
        return ops

    def get_is_cell(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        iscell = [plane['iscell'] for plane in self.F_all if 'iscell' in plane]
        return iscell

    # Get behavioral events from mat file
    def _load_behave(self) -> None:
        matfile = get_file_with_type(".mat", self.file_dir)
        self.behave = sio.loadmat(matfile)

    def get_behave(self) -> dict:
        if not self.behave:
            self._load_behave()
        return self.behave

    def get_event_df(self) -> pd.DataFrame:
        if not self.behave:
            self._load_behave()

        if not self.event_df:
            event = self.behave["eventlog"]
            self.event_df = pd.DataFrame(
                data=event, columns=["Events", "Timestamp", "Reward"]
            )
        return self.event_df

    def get_im_ts(self) -> np.array:
        
        if self.imaging_system == "Bruker":
            """
            Returns tiff image time stamps from file saved by bruker for each plane, this must be saved in the files folder

            """
            if self.num_plane == 1:
                if len(self.im_ts) == 0:
                    for file in os.listdir(self.file_dir):
                        if file.endswith(".xml"):
                            xmlfile = self.file_dir + file
                    xmlfile = get_file_with_type(".xml", self.file_dir)
                    tree = ET.parse(xmlfile)
                    root = tree.getroot()
                    self.im_ts = np.r_[
                        [child.attrib["absoluteTime"] for child in root.iter("Frame")]
                    ].astype(float)

            elif self.num_plane > 1:
                if len(self.im_ts) == 0:
                    xmlfile = get_file_with_type(".xml", self.file_dir)
                    tree = ET.parse(xmlfile)
                    root = tree.getroot()
                    for ip in range(self.num_plane):
                        ax = []
                        for child in root.iter("Frame"):
                            if child.attrib["index"] == str(ip + 1):
                                ax.append(float(child.attrib["absoluteTime"]))
                        self.im_ts.append(ax)
            last_image_ts = self.im_ts[-1][-1]
        
        elif self.imaging_system == "INSS":
            """
            Returns tiff image time stamps from the tiff header info saved by ScamImage scope for all planes.
            
            """
            if os.path.exists(os.path.join(self.file_dir, "df_timestamps.csv")):
                df = pd.read_csv(os.path.join(self.file_dir, "df_timestamps.csv"))
            else:
                tiff_file = glob.glob(os.path.join(self.data_dir, "*.tif"))
                reader = sctiffreader(tiff_file[0])
                
                num_frames = len(reader)
                # Initialize list to store data
                data = []

                # Loop through each frame and extract metadata
                for i in range(num_frames):
                    desc = reader.description(i)  # Get description for frame i
                    metadata_dict = {}

                    # Extract relevant variables using regex
                    metadata_dict['frameNumbers'] = int(re.search(r'frameNumbers = (\d+)', desc).group(1))
                    metadata_dict['frameTimestamps_sec'] = float(re.search(r'frameTimestamps_sec = ([\d\.\-e]+)', desc).group(1))
                    # metadata_dict['acquisitionNumbers'] = int(re.search(r'acquisitionNumbers = (\d+)', desc).group(1))
                    # metadata_dict['frameNumberAcquisition'] = int(re.search(r'frameNumberAcquisition = (\d+)', desc).group(1))
                    # metadata_dict['acqTriggerTimestamps_sec'] = float(re.search(r'acqTriggerTimestamps_sec = ([\d\.\-e]+)', desc).group(1))
                    # metadata_dict['nextFileMarkerTimestamps_sec'] = float(re.search(r'nextFileMarkerTimestamps_sec = ([\d\.\-e]+)', desc).group(1))
                    # metadata_dict['endOfAcquisition'] = int(re.search(r'endOfAcquisition = (\d+)', desc).group(1))
                    # metadata_dict['endOfAcquisitionMode'] = int(re.search(r'endOfAcquisitionMode = (\d+)', desc).group(1))
                    # metadata_dict['dcOverVoltage'] = int(re.search(r'dcOverVoltage = (\d+)', desc).group(1))
                    
                    # Extract epoch timestamps (year, month, day, hour, minute, second)
                    epoch_match = re.search(r'epoch = \[([\d\s\.]+)\]', desc)
                    if epoch_match:
                        metadata_dict['epoch'] = epoch_match.group(1)

                    data.append(metadata_dict)
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(self.file_dir, "df_timestamps.csv"),index=False)
                reader.close()
                
            last_image_ts = df["frameTimestamps_sec"].iloc[-1]
            
            # Add a planeindex column to the end 
            total_planes = self.num_plane + self.num_flyback
            df["planeIndex"] = df["frameNumbers"] % total_planes
            df["planeIndex"] = df["planeIndex"].replace(0, total_planes)
            df_filtered = df[df["planeIndex"].isin(range(self.num_plane+1))]

            for plane in sorted(df_filtered["planeIndex"].unique()):
                ax = df_filtered[df_filtered["planeIndex"] == plane]["frameTimestamps_sec"].tolist()
                self.im_ts.append(ax)
        
        return self.im_ts, last_image_ts

    def get_voltages(self) -> np.array:
        """
        Returns the voltage recording df with "Time(ms)", " TTL1", and " TTL2"
            - TTL1 records for entire session (>3)
            - TTL2 records when the cue is on (>3)

        """
        if self.imaging_system == "Bruker":
            if not self.voltages:
                csv = get_file_with_type(".csv", self.file_dir)
                self.voltages = pd.read_csv(csv)
            return self.voltages


def load_population_data(animal_list, day_list, data_dir, result_dir, target_frames, pre_cue_window, framerate, subtrials=None):
    
    pop_path = os.path.join(result_dir, "populationdata.npy")
    id_path = os.path.join(result_dir, "animal_id.npy")
    cells_idx_path = os.path.join(result_dir, "cells_idx.npy")
    
    if os.path.exists(os.path.join(result_dir, "populationdata.npy")):
        return np.load(pop_path, allow_pickle=True), np.load(id_path, allow_pickle=True), np.load(cells_idx_path, allow_pickle=True)

    else:
        populationdata_list = []
        animal_id = []
        cells_idx_list = []
                    
        for a, animal in enumerate(animal_list):
            
            total_cells = 0
            print(animal)
            file_dir = os.path.join(data_dir, animal, "d"+str(day_list[a]), "files")
            rawdata = np.load(os.path.join(file_dir, "F_around_cue_zscore.npy"), allow_pickle=True)  # shape: (trial_types, ntrials, ncells, nframes)
            cells_idx = np.load(os.path.join(file_dir, "cell_idx.npy"), allow_pickle=True) # load cell index            
            total_cells= sum(len(x) for x in cells_idx)
            cells_idx_list.append(cells_idx)
            
            print(total_cells, " cells loaded from animal ", animal)
            # Trials per cue and min across cues
            n_trials = [len(rawdata[i]) for i in range(rawdata.shape[0])]
            min_trials = np.min(n_trials)
            rng = np.random.default_rng() 
            
            # Build per-cue indices based on subtrials
            idxs_per_cue = []

            if subtrials is None or subtrials == 'all':
                # Use K = min trials; randomly pick K from larger cues
                K = min_trials
                for n in n_trials:
                    if n == K:
                        idx = np.arange(n, dtype=int)
                    else:
                        idx = rng.choice(n, size=K, replace=False)
                        idx = np.sort(idx)  # keep original order of the chosen trials
                    idxs_per_cue.append(idx)

            elif subtrials == 'first10':
                K = min(10, min_trials)
                for n in n_trials:
                    idxs_per_cue.append(np.arange(min(n, K), dtype=int))

            elif subtrials == 'last10':
                K = min(10, min_trials)
                for n in n_trials:
                    start = max(0, n - K)
                    idxs_per_cue.append(np.arange(start, n, dtype=int))
            
            subset_raw = [np.take(a, idx, axis=0) for a, idx in zip(rawdata, idxs_per_cue)]
            
            # Baseline subtraction before averaging
            # baseline = np.mean(subset_raw[:, :, :, 0:int(pre_cue_window * framerate)], axis=3, keepdims=True)
            # subset_baseline_subtract = subset_raw - baseline                

            # Average across trials within each trial type
            subset_ave = np.mean(subset_raw, axis=1)  # shape: (trial_types, ncells, nframes)
            
            # Baseline subtraction after averaging
            baseline = np.mean(subset_ave[:, :, 0:pre_cue_window *framerate], axis=2, keepdims=True)
            subset_ave = subset_ave - baseline  # baseline subtraction  
            
            tempdata = subset_ave.transpose(1, 0, 2).reshape(subset_ave.shape[1], -1) # reshape into (ncells, trial_types*nframes)
            ncells, nframes = tempdata.shape
            assert(ncells == total_cells)
            
            if nframes < target_frames:
                pad_width = target_frames - nframes
                tempdata = np.pad(tempdata, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            elif nframes > target_frames:
                tempdata = tempdata[:, :target_frames]
            
            # # Perform baseline subtraction of the 3s pre CS period
            # baseline_window = int(pre_cue_window * framerate)
            # for i in range(len(trial_types)):
            #     start = i * window_size
            #     end = start + window_size
            #     baseline = np.mean(tempdata[:,  start:start + baseline_window], axis=1, keepdims=True)
            #     tempdata[:, start:end] -= baseline
            populationdata_list.append(tempdata)
            animal_id.extend([animal] * ncells) # track neuron to specific animal
        
        populationdata  = np.vstack(populationdata_list) # shape: (total_ncells, trial_types*window_size)
        animal_id = np.array(animal_id)
        np.save(pop_path, populationdata)
        np.save(id_path, animal_id)
        np.save(cells_idx_path, cells_idx_list)
        
        return populationdata, animal_id, cells_idx_list
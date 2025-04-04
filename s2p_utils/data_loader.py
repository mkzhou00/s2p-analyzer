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

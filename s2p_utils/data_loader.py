import logging
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import xml.etree.ElementTree as ET

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
    def __init__(self, data_dir: str, num_planes: int) -> None:
        """
        Data set inclduing Suite2p, behavioral, and image ts. All the files
        are lazy loaded upon request.

        Args:
            data_dir: Data directory containing all the required files.
        """
        self.s2p_dir = os.path.join(data_dir, "suite2p")
        self.file_dir = os.path.join(data_dir, "files")
        self.plane_subfolder = "plane"
        self.num_plane = num_planes

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
                temp = []
                temp = sio.loadmat(
                    os.path.join(
                        self.s2p_dir, self.plane_subfolder + str(ip), "Fall.mat"
                    )
                )
                self.F_all.append(temp)
        return self.F_all

    def get_F(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        F = []
        for plane in self.F_all:
            F.append(plane['F'])
        return F

    def get_Fneu(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        Fneu = []
        for plane in self.F_all:
            Fneu.append(plane['Fneu'])
        return Fneu

    def get_spks(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        spks = []
        for plane in self.F_all:
            spks.append(plane['spks'])
        return spks

    def get_stat(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        stat = []
        for plane in self.F_all:
            stat.append(plane['stat'])
        return stat

    def get_ops(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        ops = []
        for plane in self.F_all:
            ops.append(plane['ops'])
        return ops

    def get_is_cell(self) -> np.array:
        if len(self.F_all) == 0:
            self._load_F_all()

        iscell = []
        for plane in self.F_all:
            iscell.append(plane['iscell'])
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
        """
        Returns tiff image time stamps from 000.xml file saved by bruker for each plane.

        Note: Need to change the actual xml name containing time points information if it ends with something else other than 000.xml

        """
        if self.num_plane == 1:
            if len(self.im_ts) == 0:
                for file in os.listdir(self.file_dir):
                    if file.endswith("000.xml"):
                        xmlfile = self.file_dir + file
                xmlfile = get_file_with_type(".xml", self.file_dir)
                tree = ET.parse(xmlfile)
                root = tree.getroot()
                self.im_ts = np.r_[
                    [child.attrib["absoluteTime"] for child in root.iter("Frame")]
                ].astype(float)

        elif self.num_plane > 1:
            if len(self.im_ts) == 0:
                xmlfile = get_file_with_type("000.xml", self.file_dir)
                tree = ET.parse(xmlfile)
                root = tree.getroot()
                for ip in range(self.num_plane):
                    ax = []
                    for child in root.iter("Frame"):
                        if child.attrib["index"] == str(ip + 1):
                            ax.append(float(child.attrib["absoluteTime"]))
                    self.im_ts.append(ax)

        return self.im_ts

    def get_voltages(self) -> np.array:
        """
        Returns the voltage recording df with "Time(ms)", " TTL1", and " TTL2"
            - TTL1 records for entire session (>3)
            - TTL2 records when the cue is on (>3)

        """
        if not self.voltages:
            csv = get_file_with_type(".csv", self.file_dir)
            self.voltages = pd.read_csv(csv)
        return self.voltages

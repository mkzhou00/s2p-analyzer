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
    def __init__(self, data_dir: str) -> None:
        """
        Data set inclduing Suite2p, behavioral, and image ts. All the files
        are lazy loaded upon request.

        Args:
            data_dir: Data directory containing all the required files.
        """
        self.dir = data_dir

        # Suite2p output.
        self.F = None
        self.Fneu = None
        self.spks = None
        self.stat = None
        self.ops = None
        self.is_cell = None

        # Behavioral data. (mat)
        self.behave = None
        self.event_df = None

        # Image timestamps. (xml)
        self.im_ts = None

        # Session start and end timestamps. (csv)
        self.voltages = None

    def get_F(self) -> np.array:
        if not self.F:
            self.F = np.load(os.path.join(self.dir, "F.npy"), allow_pickle=True)
        return self.F

    def get_Fneu(self) -> np.array:
        if not self.Fneu:
            self.Fneu = np.load(os.path.join(self.dir, "Fneu.npy"), allow_pickle=True)
        return self.Fneu

    def get_spks(self) -> np.array:
        if not self.spks:
            self.spks = np.load(os.path.join(self.dir, "spks.npy"), allow_pickle=True)
        return self.spks

    def get_stat(self) -> np.array:
        if not self.stat:
            self.stat = np.load(os.path.join(self.dir, "stat.npy"), allow_pickle=True)
        return self.stat

    def get_ops(self) -> np.array:
        if not self.ops:
            self.ops = np.load(os.path.join(self.dir, "ops.npy"), allow_pickle=True)
        return self.ops

    def get_is_cell(self) -> np.array:
        if not self.is_cell:
            self.is_cell = np.load(
                os.path.join(self.dir, "iscell.npy"), allow_pickle=True
            )
        return self.is_cell

    def _load_behave(self) -> None:
        matfile = get_file_with_type(".mat", self.dir)
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
        if not self.im_ts:
            xmlfile = get_file_with_type(".xml", self.dir)
            tree = ET.parse(xmlfile)
            root = tree.getroot()
            self.im_ts = np.r_[
                [child.attrib["absoluteTime"] for child in root.iter("Frame")]
            ].astype(float)

        return self.im_ts

    def get_voltages(self) -> np.array:
        if not self.voltages:
            csv = get_file_with_type(".csv", self.dir)
            self.voltages = pd.read_csv(csv)
        return self.voltages

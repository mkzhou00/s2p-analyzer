import argparse
import logging

from s2p_utils import data_loader


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
        type=int,
        default=0.5,
        help="Minimum probability to identify an ROI as a cell.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=".",
        help="Output Folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    main()

"""Combined Dataset Loader for 360° Video Multimodal Viewport Prediction.

STRICT DATASET REQUIREMENT:
- Loads ONLY the 'combined.csv' dataset file for any person-video pair.
- NEVER loads, references, or falls back to separate head.csv, eye.csv, or face.csv files.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default root directory where DataCollectedVR is located
DEFAULT_DATA_ROOT = PROJECT_ROOT / "DataCollectedVR"

# Standard video mappings
VIDEO_NAMES_MAP: Dict[str, str] = {
    "V01": "video_01_Weekly_Idol-Dancing",
    "V02": "video_02_Damai_Music_Live-Vocie_Toy",
    "V03": "video_03_Rio_Olympics_VR_Interview",
    "V04": "video_04_Female_Basketball_Match",
    "V05": "video_05_SHOWTIME_Boxing",
    "V06": "video_06_Rosa_Khutor_Ski_360",
    "V07": "video_07_Hammerhead_Shark_NatGeo_360",
    "V08": "video_08_Freestyle_Skiing",
    "V09": "video_09_Google_Spotlight-HELP",
    "V10": "video_10_Deep_Ocean_Horror_360",
    "V11": "video_11_GoPro_VR-Tahiti_Surf",
    "V12": "video_12_LOSC_Football",
    "V13": "video_13_Jungle_Book_360",
    "V14": "video_14_Death_Note_VR",
}

EXPECTED_COLUMNS: List[str] = [
    "Timestamp",
    "PlaybackTime",
    "VideoFrame",
    "HeadYaw",
    "HeadPitch",
    "HeadRoll",
    "GazeYaw",
    "GazePitch",
    "GazeRelativeH",
    "GazeRelativeV",
    "EyeHeadOffset",
    "AbsoluteGazeYaw",
    "AbsoluteGazePitch",
    "LeftOpenness",
    "RightOpenness",
    "BlinkL",
    "BlinkR",
    "FaceIsValid",
    "SmileL",
    "SmileR",
    "BrowDownL",
    "BrowDownR",
    "JawOpen",
    "EyeClosedL",
    "EyeClosedR",
    "FaceActivity",
]


class CombinedDatasetLoader:
    """Loader dedicated strictly to the Combined dataset across all 25 persons and 14 videos."""

    def __init__(self, data_root: Union[str, Path] = DEFAULT_DATA_ROOT):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {self.data_root}")
        self._person_dir_cache: Dict[str, Path] = {}
        self._df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._discover_person_directories()

    def _discover_person_directories(self) -> None:
        """Discovers and caches the folder paths for persons P01 to P25 (P001 to P025)."""
        if not self.data_root.is_dir():
            return

        for entry in self.data_root.iterdir():
            if entry.is_dir() and entry.name.startswith("P"):
                # e.g., P001-20260710T102047Z-2-001
                prefix = entry.name.split("-")[0]  # P001
                num_part = prefix.lstrip("P")  # '001'
                try:
                    num_int = int(num_part)
                    p_id_2digit = f"P{num_int:02d}"  # P01
                    p_id_3digit = f"P{num_int:03d}"  # P001

                    # Look inside for inner P001 folder
                    inner_candidates = [
                        sub for sub in entry.iterdir() if sub.is_dir() and sub.name.startswith("P")
                    ]
                    actual_person_dir = inner_candidates[0] if inner_candidates else entry

                    self._person_dir_cache[p_id_2digit] = actual_person_dir
                    self._person_dir_cache[p_id_3digit] = actual_person_dir
                except ValueError:
                    continue

    def list_available_persons(self) -> List[str]:
        """Returns sorted list of 2-digit person identifiers: P01 to P25."""
        p_ids = set()
        for k in self._person_dir_cache.keys():
            if len(k) == 3:  # 'P01' format
                p_ids.add(k)
        return sorted(p_ids)

    def list_available_videos(self) -> List[Tuple[str, str]]:
        """Returns sorted list of (video_id, video_folder_name) tuples."""
        return sorted(VIDEO_NAMES_MAP.items())

    def normalize_person_id(self, person_id: str) -> str:
        """Normalizes any person identifier (e.g. '1', 'P1', 'P01', 'P001') to 'P01'."""
        cleaned = person_id.strip().upper()
        if cleaned.startswith("P"):
            digits = cleaned[1:]
        else:
            digits = cleaned
        num = int(digits)
        return f"P{num:02d}"

    def normalize_video_id(self, video_id: str) -> Tuple[str, str]:
        """Normalizes video ID (e.g. 'V1', 'V01', 'video_01') to ('V01', full_folder_name)."""
        cleaned = video_id.strip()
        if cleaned in VIDEO_NAMES_MAP:
            return cleaned, VIDEO_NAMES_MAP[cleaned]

        cleaned_upper = cleaned.upper()
        if cleaned_upper in VIDEO_NAMES_MAP:
            return cleaned_upper, VIDEO_NAMES_MAP[cleaned_upper]

        # Check by folder name
        for v_code, v_folder in VIDEO_NAMES_MAP.items():
            if cleaned == v_folder or cleaned.startswith(f"video_{v_code[1:]}"):
                return v_code, v_folder

        # Check by digit
        num_str = "".join(ch for ch in cleaned if ch.isdigit())
        if num_str:
            v_code = f"V{int(num_str):02d}"
            if v_code in VIDEO_NAMES_MAP:
                return v_code, VIDEO_NAMES_MAP[v_code]

        raise ValueError(f"Unrecognized video identifier: '{video_id}'")

    def resolve_combined_path(self, person_id: str, video_id: str) -> Path:
        """Resolves the absolute path to combined.csv for the given person and video.

        STRICT: Only combined.csv is ever returned.
        """
        norm_p = self.normalize_person_id(person_id)
        norm_v_code, norm_v_folder = self.normalize_video_id(video_id)

        if norm_p not in self._person_dir_cache:
            raise FileNotFoundError(
                f"Person '{person_id}' (normalized: {norm_p}) not found in dataset."
            )

        person_dir = self._person_dir_cache[norm_p]
        video_dir = person_dir / norm_v_folder
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

        combined_file = video_dir / "combined.csv"
        if not combined_file.exists():
            raise FileNotFoundError(
                f"STRICT CHECK FAILED: Combined dataset file not found at: {combined_file}"
            )

        return combined_file

    def load_combined_dataset(
        self, person_id: str, video_id: str, validate_columns: bool = True
    ) -> pd.DataFrame:
        """Loads and strictly validates the combined dataset for a specific person and video."""
        norm_p = self.normalize_person_id(person_id)
        norm_v_code, _ = self.normalize_video_id(video_id)
        cache_key = (norm_p, norm_v_code)
        if cache_key in self._df_cache:
            return self._df_cache[cache_key].copy()

        file_path = self.resolve_combined_path(person_id, video_id)
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(f"Combined dataset at {file_path} is empty (0 rows).")

        if validate_columns:
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Combined dataset at {file_path} is missing expected columns: {missing}"
                )

        # Enforce chronological ordering on Timestamp
        if not df["Timestamp"].is_monotonic_increasing:
            df = df.sort_values("Timestamp").reset_index(drop=True)

        self._df_cache[cache_key] = df
        return df.copy()

    def get_all_records_inventory(self) -> List[Dict[str, Union[str, int, bool]]]:
        """Scans all 25 persons x 14 videos (350 combinations) and verifies combined.csv existence."""
        inventory = []
        persons = self.list_available_persons()
        videos = self.list_available_videos()

        for p in persons:
            for v_code, v_name in videos:
                entry = {
                    "person_id": p,
                    "video_id": v_code,
                    "video_name": v_name,
                    "exists": False,
                    "file_path": "",
                    "file_size_bytes": 0,
                }
                try:
                    cpath = self.resolve_combined_path(p, v_code)
                    entry["exists"] = True
                    entry["file_path"] = str(cpath)
                    entry["file_size_bytes"] = cpath.stat().st_size
                except Exception as ex:
                    entry["error"] = str(ex)
                inventory.append(entry)

        return inventory


if __name__ == "__main__":
    loader = CombinedDatasetLoader()
    print(f"Discovered {len(loader.list_available_persons())} persons.")
    print(f"Discovered {len(loader.list_available_videos())} videos.")
    inv = loader.get_all_records_inventory()
    present_cnt = sum(1 for item in inv if item["exists"])
    print(f"Total Combined files present: {present_cnt} / {len(inv)}")
    # Sample load
    sample_df = loader.load_combined_dataset("P01", "V01")
    print(f"Successfully loaded P01 V01 combined.csv with shape: {sample_df.shape}")

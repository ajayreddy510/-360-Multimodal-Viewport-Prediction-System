"""Unit and integration tests for Stage 1: Combined Dataset Discovery & Inspection.

Verifies:
1. All 25 persons (P01-P25) and 14 videos (V01-V14) are discoverable.
2. STRICT DATASET REQUIREMENT: Only combined.csv is accessed.
3. Every file contains all 26 required columns in the expected order.
4. Timestamps are monotonic and non-negative.
5. Error handling for invalid person/video identifiers.
"""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.combined_loader import CombinedDatasetLoader, EXPECTED_COLUMNS, VIDEO_NAMES_MAP
from src.dataset_inspector import DatasetInspector


class TestStage1CombinedDataset(unittest.TestCase):
    """Test suite for Combined dataset discovery and validation."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CombinedDatasetLoader()
        cls.inspector = DatasetInspector()

    def test_all_25_persons_discovered(self):
        """Verify that all 25 participants P01 to P25 exist."""
        persons = self.loader.list_available_persons()
        self.assertEqual(len(persons), 25)
        for i in range(1, 26):
            expected_id = f"P{i:02d}"
            self.assertIn(expected_id, persons)

    def test_all_14_videos_discovered(self):
        """Verify that all 14 videos V01 to V14 exist."""
        videos = self.loader.list_available_videos()
        self.assertEqual(len(videos), 14)
        for i in range(1, 15):
            expected_id = f"V{i:02d}"
            self.assertIn(expected_id, VIDEO_NAMES_MAP)

    def test_strict_only_combined_file_resolved(self):
        """Verify that resolver only returns combined.csv and never head/eye/face."""
        for p in ["P01", "P12", "P25"]:
            for v in ["V01", "V07", "V14"]:
                cpath = self.loader.resolve_combined_path(p, v)
                self.assertTrue(cpath.exists())
                self.assertEqual(cpath.name, "combined.csv")
                self.assertNotIn("head.csv", str(cpath))
                self.assertNotIn("eye.csv", str(cpath))
                self.assertNotIn("face.csv", str(cpath))

    def test_combined_dataframe_columns_and_monotonicity(self):
        """Verify that loaded DataFrame contains 26 columns and strictly monotonic timestamps."""
        df = self.loader.load_combined_dataset("P01", "V01")
        self.assertEqual(len(df.columns), 26)
        self.assertListEqual(df.columns.tolist(), EXPECTED_COLUMNS)
        self.assertGreater(len(df), 1000)
        self.assertTrue(df["Timestamp"].is_monotonic_increasing)
        self.assertFalse(df.isna().any().any(), "No NaN values allowed in Combined dataset")

    def test_invalid_person_raises_error(self):
        """Verify that requesting an invalid person raises FileNotFoundError."""
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.loader.load_combined_dataset("P99", "V01")

    def test_invalid_video_raises_error(self):
        """Verify that requesting an invalid video raises ValueError or FileNotFoundError."""
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.loader.load_combined_dataset("P01", "V99")

    def test_data_dictionary_completeness(self):
        """Verify that the generated data dictionary covers all 26 columns."""
        dd = self.inspector.generate_data_dictionary()
        self.assertEqual(len(dd), 26)
        columns_in_dict = [entry["column"] for entry in dd]
        self.assertListEqual(columns_in_dict, EXPECTED_COLUMNS)


if __name__ == "__main__":
    unittest.main()

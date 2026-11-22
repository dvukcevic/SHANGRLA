import pytest
import sys
from pathlib import Path

from shangrla.raire.raire_utils import load_contests_from_raire

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"

class TestInput:
  
  def test_load_raire_file(self):
      data_file = TEST_DATA_DIR / "Aspen_2009_Mayor.raire"
      assert len(load_contests_from_raire(data_file)) == 2

if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))

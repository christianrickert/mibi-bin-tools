import pytest
from pytest_cases import parametrize, parametrize_with_cases, case
from typing import Dict, Tuple
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from mibi_bin_tools import bin_files

THIS_DIR = Path(__file__).parent

TEST_DATA_DIR = THIS_DIR / 'data'


class FovMetadataTestPanels:

    def case_global_panel_success(self):
        return (-0.3, 0.3)

    def case_specified_panel_success(self):
        panel = pd.DataFrame([{
            'Mass': 89,
            'Target': 'HH3',
            'Start': 88.7,
            'Stop': 89.0,
        }])
        return panel
   
    def case_specified_panel_failure(self):
        bad_panel = pd.DataFrame([{
            'isotope': 89,
            'antibody': 'HH3',
            'start': 88.7,
            'stop': 89,
        }])
        return bad_panel


class FovMetadataTestChannels:

    def case_no_channel_filter_success(self):
        return None

    def case_channel_filter_success(self):
        return ['HH3']
    
    def case_channel_filter_failure(self):
        return ['HH2']


class FovMetadataTestIntensities:

    @parametrize(('do_all_intensities',), (True, False))
    def case_global_intensities_success(self, do_all_intensities):
        return do_all_intensities

    def case_specified_intensities_success(self):
        return ['HH3']

    def case_format_intensities_failure(self):
        return ['HH2']


def test_write_out():

    img_data = np.zeros((3, 10, 10, 5), dtype=np.uint32)
    fov_name = 'fov1'
    targets = [chr(ord('a') + i) for i in range(5)]

    inner_dir_names = [
        '',
        'intensities',
        'intensity_times_width',
    ]

    suffix_names = [
        '',
        '_intensity',
        '_int_width',
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # correctness
        bin_files._write_out(img_data, tmpdir, fov_name, targets)

        assert(os.path.exists(os.path.join(tmpdir, fov_name)))
        for inner_name, suffix in zip(inner_dir_names, suffix_names):
            inner_dir = os.path.join(tmpdir, fov_name, inner_name)
            assert(os.path.exists(inner_dir))
            for target in targets:
                assert(os.path.exists(os.path.join(inner_dir, f'{target}{suffix}.tiff')))

    pass


def _make_blank_file(folder: str, name: str):
    with open(os.path.join(folder, name), 'w') as f:
        pass


def test_find_bin_files():

    files: Dict[str, Tuple[bool, bool]] = {
        'fov1': (True, True),
        'fov2': (True, True),
        'fov3': (True, True),
        'fov4': (True, False),
        'fov5': (False, True)
    }

    include_fovs = ['fov1', 'fov2']

    with tempfile.TemporaryDirectory() as tmpdir:
        # create test environment
        for fov_name, (make_bin, make_json) in files.items():
            if make_bin:
                _make_blank_file(tmpdir, f'{fov_name}.bin')
            if make_json:
                _make_blank_file(tmpdir, f'{fov_name}.json')

        # correctness
        fov_dict = bin_files._find_bin_files(tmpdir)
        assert(set(fov_dict.keys()) == {'fov1', 'fov2', 'fov3'})

        # include_fovs check
        fov_dict = bin_files._find_bin_files(tmpdir, include_fovs=include_fovs)
        assert(set(fov_dict.keys()) == set(include_fovs))


@parametrize_with_cases('panel', cases=FovMetadataTestPanels, glob='*_success')
@parametrize_with_cases('channels', cases=FovMetadataTestChannels, glob='*_success')
@parametrize_with_cases('intensities', cases=FovMetadataTestIntensities, glob='*_success')
def test_fill_fov_metadata_success(panel, channels, intensities):
    fov = {
        'json': 'fov-1-scan-1.json',
        'bin': 'fov-1-scan-1.bin',
    }

    time_res = 0.5
    # panel type can vary (test intensities)
    bin_files._fill_fov_metadata(TEST_DATA_DIR, fov, panel, intensities, time_res, channels)
    pass


# TODO: get reasonable sized test data for this
def test_extract_bin_files():
    pass


# TODO: get reasonable sized test data for this
def test_get_width_histogram():
    pass


# TODO: get resonalbe sized test data for this
def test_median_height_vs_mean_pp():
    pass

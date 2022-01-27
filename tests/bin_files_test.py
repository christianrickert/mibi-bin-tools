from pytest_cases import (
    parametrize, parametrize_with_cases, fixture
)
import pytest
from typing import Dict, Tuple
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from mibi_bin_tools import bin_files, type_utils, _extract_bin

THIS_DIR = Path(__file__).parent

TEST_DATA_DIR = THIS_DIR / 'data'


class FovMetadataTestFiles:

    def _generic(self, parent_folder):
        return os.path.join(TEST_DATA_DIR, parent_folder), {
            'json': 'fov-1-scan-1.json',
            'bin': 'fov-1-scan-1.bin',
        }

    def case_tissue(self):
        return self._generic('tissue')

    def case_moly(self):
        return self._generic('moly')


class FovMetadataTestPanels:

    def case_global_panel_success(self):
        return (-0.3, 0.3)

    def case_global_panel_failure(self):
        return (-0.3, 0.3), KeyError

    def case_specified_panel_success(self):
        panel = pd.DataFrame([{
            'Mass': 89,
            'Target': 'SMA',
            'Start': 88.7,
            'Stop': 89.0,
        }])
        return panel

    def case_specified_panel_failure(self):
        bad_panel = pd.DataFrame([{
            'isotope': 89,
            'antibody': 'SMA',
            'start': 88.7,
            'stop': 89,
        }])
        return bad_panel, KeyError


class FovMetadataTestChannels:

    def case_no_channel_filter_success(self):
        return None

    def case_channel_filter_success(self):
        return ['SMA']

    def case_channel_filter_failure(self):
        return ['HH2']


class FovMetadataTestIntensities:

    @parametrize(('do_all_intensities',), (True, False))
    def case_global_intensities_success(self, do_all_intensities):
        return do_all_intensities

    def case_specified_intensities_success(self):
        return ['SMA']

    def case_format_intensities_failure(self):
        return ['HH2']


@fixture
def filepath_checks():
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

    def _filepath_checks(out_dir, fov_name, targets, intensities):
        assert(os.path.exists(os.path.join(out_dir, fov_name)))
        for i, (inner_name, suffix) in enumerate(zip(inner_dir_names, suffix_names)):
            inner_dir = os.path.join(out_dir, fov_name, inner_name)
            made_intensity_folder = i < 1 or type_utils.any_true(intensities)
            if made_intensity_folder:
                assert(os.path.exists(inner_dir))
            else:
                assert(not os.path.exists(inner_dir))
            for target in targets:
                tif_path = os.path.join(inner_dir, f'{target}{suffix}.tiff')
                if made_intensity_folder:
                    assert(os.path.exists(tif_path))
                else:
                    assert(not os.path.exists(tif_path))

    return _filepath_checks


def test_write_out(filepath_checks):

    img_data = np.zeros((3, 10, 10, 5), dtype=np.uint32)
    fov_name = 'fov1'
    targets = [chr(ord('a') + i) for i in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        # correctness
        bin_files._write_out(img_data, tmpdir, fov_name, targets)
        filepath_checks(tmpdir, fov_name, targets, True)

    pass


def _make_blank_file(folder: str, name: str):
    with open(os.path.join(folder, name), 'w'):
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

        with pytest.raises(FileNotFoundError, match='No viable bin file'):
            fov_dict = bin_files._find_bin_files(tmpdir, include_fovs=['fov_fake'])


@parametrize_with_cases('test_dir, fov', cases=FovMetadataTestFiles, glob='tissue')
@parametrize_with_cases('panel', cases=FovMetadataTestPanels, glob='*_success')
@parametrize_with_cases('channels', cases=FovMetadataTestChannels, glob='*_success')
@parametrize_with_cases('intensities', cases=FovMetadataTestIntensities, glob='*_success')
def test_fill_fov_metadata_success(test_dir, fov, panel, channels, intensities):
    time_res = 0.5
    # panel type can vary (test intensities)
    bin_files._fill_fov_metadata(test_dir, fov, panel, intensities, time_res, channels)
    pass


@parametrize_with_cases('test_dir, fov', cases=FovMetadataTestFiles, glob='moly')
@parametrize_with_cases('panel, err', cases=FovMetadataTestPanels, glob='*_failure')
@parametrize_with_cases('channels', cases=FovMetadataTestChannels, glob='*_success')
@parametrize_with_cases('intensities', cases=FovMetadataTestIntensities, glob='*_success')
def test_fill_fov_metadata_failure(test_dir, fov, panel, err, channels, intensities):
    time_res = 0.5
    with pytest.raises(err):
        bin_files._fill_fov_metadata(test_dir, fov, panel, intensities, time_res, channels)


@parametrize_with_cases('test_dir, fov', cases=FovMetadataTestFiles)
@parametrize_with_cases('panel', cases=FovMetadataTestPanels, glob='specified_panel_success')
@parametrize_with_cases('intensities', cases=FovMetadataTestIntensities, glob='*_success')
def test_extract_bin_files(test_dir, fov, panel, intensities, filepath_checks):
    time_res = 500e-6
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_files.extract_bin_files(test_dir, tmpdir, None, panel, intensities, time_res)
        filepath_checks(tmpdir, fov['json'].split('.')[0], panel['Target'].values, intensities)


@parametrize_with_cases('test_dir, fov', cases=FovMetadataTestFiles)
@parametrize_with_cases('panel', cases=FovMetadataTestPanels, glob='specified_panel_success')
def test_get_width_histogram(test_dir, fov, panel):
    bin_files.get_histograms_per_tof(
        test_dir,
        fov['json'].split('.')[0],
        'SMA',
        panel,
        time_res=500e-6
    )


@parametrize_with_cases('test_dir, fov', cases=FovMetadataTestFiles)
@parametrize_with_cases('panel', cases=FovMetadataTestPanels, glob='specified_panel_success')
def test_median_height_vs_mean_pp(test_dir, fov, panel):
    bin_files.median_height_vs_mean_pp(
        test_dir,
        fov['json'].split('.')[0],
        'SMA',
        panel,
        500e-6
    )
    pass


@parametrize_with_cases('test_dir, fov', cases=FovMetadataTestFiles)
@parametrize_with_cases('panel', cases=FovMetadataTestPanels, glob='full_range')
def test_get_total_counts(test_dir, fov):
    total_counts = bin_files.get_total_counts(test_dir)

    bf = os.path.join(test_dir, fov['bin'])
    total_ion_image = _extract_bin.c_extract_bin(
        bytes(bf, 'utf-8'), np.array([0], np.uint16),
        np.array([-1], dtype=np.uint16), np.array([False], dtype=np.uint8)
    )
    assert(total_counts['fov-1-scan-1'] == np.sum(total_ion_image[0, :, :, :]))

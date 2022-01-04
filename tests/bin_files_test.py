import pytest
import numpy as np

from mibi_bin_tools import bin_files

def test_mass2tof():
    mass_array = np.array([0, 1, 2])
    time_res = 1
    mass_offset = 0
    mass_gain = 1

    # test for correctness
    time_array = bin_files._mass2tof(mass_array, mass_offset, mass_gain, time_res)
    np.testing.assert_array_equal(np.sqrt(mass_array), time_array)

    # test mass offset change
    time_array = bin_files._mass2tof(mass_array, 1, mass_gain, time_res)
    np.testing.assert_array_equal(np.sqrt(mass_array) + 1, time_array)

    # test mass gain change
    time_array = bin_files._mass2tof(mass_array, mass_offset, 2, time_res)
    np.testing.assert_array_equal(2 * np.sqrt(mass_array), time_array)

    # test time_res change
    time_array = bin_files._mass2tof(mass_array, mass_offset, mass_gain, 0.5)
    np.testing.assert_array_equal(2 * np.sqrt(mass_array), time_array)

def test_set_tof_ranges():
    

    pass

def test_write_out():
    pass

def test_find_bin_files():
    pass

def test_fill_fov_metadata():
    pass

def test_parse_global_panel():
    pass

def test_parse_df_panel():
    pass

def test_parse_intensities():
    pass

def test_extract_bin_files():
    pass

def test_get_width_histogram():
    pass

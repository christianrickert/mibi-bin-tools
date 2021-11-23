from typing import Any, Dict, List, Tuple, Union
import os
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import skimage.io as io

from mibi_bin_tools import io_utils, _extract_bin

def mass2tof(masses_arr, mass_offset, mass_gain, time_res):
    return (mass_gain * np.sqrt(masses_arr) + mass_offset) / time_res

def _set_tof_ranges(fov: Dict[str, Any], higher: np.ndarray, lower: np.ndarray, time_res: float):
    key_names = ('upper_tof_range', 'lower_tof_range')
    mass_ranges = (higher, lower)
    wrapping_functions = (np.ceil, np.floor)

    for key, masses, wrap in zip(key_names, mass_ranges, wrapping_functions):
        fov[key] = \
            wrap(
                mass2tof(masses, fov['mass_offset'], fov['mass_gain'], time_res)
            ).astype(np.uint16)

def write_out(img_data, out_dir, fov_name, targets):
    out_dirs = [
        os.path.join(out_dir, fov_name),
        os.path.join(out_dir, fov_name, 'intensities'),
        os.path.join(out_dir, fov_name, 'intensity_times_width')
    ]
    suffixes = [
        '',
        '_intensity',
        '_int_width'
    ]
    for i, out_dir, suffix in enumerate(zip(out_dirs, suffixes)):
        if not os.path.exitst(out_dir):
            os.makedirs(out_dir)
        for j, target in enumerate(targets):
            io.imsave(os.path.join(out_dir, f'{target}{suffix}.tiff'), img_data[i, :, :, j], plugin='tifffile', check_contrast=False)

def extract_bin_files(data_dir: str, out_dir: str,
                      include_fovs: Union[List[str], None] = None,
                      panel: Union[Tuple[float, float], pd.DataFrame] = (-0.3, 0.0),
                      intensities: Union[bool, List[str]] = False, time_res: float=500e-6):
    
    # TODO: intensities

    bin_files = io_utils.list_files(data_dir, substrs=['.bin'])
    json_files = io_utils.list_files(data_dir, substrs=['.json'])

    fov_names = io_utils.extract_delimited_names(bin_files, delimiter='.')

    fov_files = {
        fov_name: {
            'bin': fov_name + '.bin',
            'json': fov_name + '.json',
        }
        for fov_name in fov_names
        if fov_name + '.json' in json_files
    }

    if include_fovs is not None:
        fov_files = {fov_file: fov_files[fov_file] for fov_file in include_fovs}

    for fov in fov_files.values():
        with open(os.path.join(data_dir, fov['json']), 'rb') as f:
            data = json.load(f)

        fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
        fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']

        if type(panel) is tuple:
            rows = data['fov']['panel']['conjugates']
            fov['masses'], fov['targets'] = zip(*[(el['mass'], el['target']) for el in rows])

            masses_arr = np.array(fov['masses'])
            _set_tof_ranges(fov, masses_arr + panel[1], masses_arr + panel[0], time_res)
        else:
            fov['masses'] = panel['Mass']
            fov['targets'] = panel['Target']

            _set_tof_ranges(fov, panel['Stop'].values, panel['Start'].values, time_res)

        if type(intensities) is list:
            fov['intensities'] = [target in intensities for target in fov['targets']]
        elif intensities is True:
            fov['intensities'] = fov['targets']

        # order the 'calc_intensity' bools 
        if 'intensities' in fov.keys():
            fov['calc_intensity'] = [target in fov['intensities'] for target in fov['targets']]
        else:
            fov['calc_intensity'] = [False,] * len(fov['targets'])

    # start download of bin files to new tmp dir
    bin_file_paths = [os.path.join(data_dir, fov['bin']) for fov in fov_files.values()]
    bin_files = \
        [(fov, os.path.join(data_dir, fov['bin'])) for fov in fov_files.values()]

    with mp.Pool() as pool:
        for i, (fov, bf) in enumerate(bin_files):
            # call extraction cython here
            img_data = _extract_bin.c_extract_bin(
                bytes(bf, 'utf-8'), fov['lower_tof_range'],
                fov['upper_tof_range'], np.array(fov['calc_intensity'], dtype=np.uint8))
            pool.apply_async(
                write_out, 
                (img_data, out_dir, fov['bin'][:-4], fov['targets'])
            )
        pool.close()
        pool.join()

def extract_no_sum(data_dir, out_dir, fov, channel, mass_range=(-0.3, 0.0), time_res: float=500e-6):
    bin_files = io_utils.list_files(data_dir, substrs=['.bin'])
    json_files = io_utils.list_files(data_dir, substrs=['.json'])

    fov_names = io_utils.extract_delimited_names(bin_files, delimiter='.')

    fov_files = {
        fov_name: {
            'bin': fov_name + '.bin',
            'json': fov_name + '.json',
        }
        for fov_name in fov_names
        if fov_name + '.json' in json_files
    }

    fov = fov_files[fov]
    with open(os.path.join(data_dir, fov['json']), 'rb') as f:
        data = json.load(f)

    fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
    fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']

    rows = data['fov']['panel']['conjugates']
    fov['masses'], fov['targets'] = zip(*[(el['mass'], el['target']) for el in rows])
    t_index = fov['targets'].index(channel)
    fov['masses'] = fov['masses'][t_index]
    fov['targets'] = fov['targets'][t_index]

    masses_arr = np.array(fov['masses'])
    _set_tof_ranges(fov, masses_arr + mass_range[1], masses_arr + mass_range[0], time_res)

    # start download of bin files to new tmp dir
    bin_file_paths = os.path.join(data_dir, fov['bin'])

    local_bin_file = os.path.join(data_dir, fov['bin'])

    with mp.Pool() as pool:
        # call extraction cython here
        discovered = _extract_bin.c_extract_no_sum(bytes(local_bin_file, 'utf-8'), 
                                                   fov['lower_tof_range'],
                                                   fov['upper_tof_range'] 
        )
        pool.close()
        pool.join()

    return discovered


def median_height_vs_mean_pp(data_dir, fov, channel, mass_range=(-0.3, 0.0),
                             time_res: float=500e-6):
    bin_files = io_utils.list_files(data_dir, substrs=['.bin'])
    json_files = io_utils.list_files(data_dir, substrs=['.json'])

    fov_names = io_utils.extract_delimited_names(bin_files, delimiter='.')

    fov_files = {
        fov_name: {
            'bin': fov_name + '.bin',
            'json': fov_name + '.json',
        }
        for fov_name in fov_names
        if fov_name + '.json' in json_files
    }

    fov = fov_files[fov]
    with open(os.path.join(data_dir, fov['json']), 'rb') as f:
        data = json.load(f)

    fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
    fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']

    rows = data['fov']['panel']['conjugates']
    fov['masses'], fov['targets'] = zip(*[(el['mass'], el['target']) for el in rows])
    t_index = fov['targets'].index(channel)
    fov['masses'] = fov['masses'][t_index]
    fov['targets'] = fov['targets'][t_index]

    masses_arr = np.array(fov['masses'])
    _set_tof_ranges(fov, masses_arr + mass_range[1], masses_arr + mass_range[0], time_res)

    local_bin_file = os.path.join(data_dir, fov['bin'])

    median_height, mean_pp = \
        _extract_bin.c_pulse_height_vs_positive_pixel(bytes(local_bin_file, 'utf-8'), 
                                                      fov['lower_tof_range'],
                                                      fov['upper_tof_range'])

    return median_height, mean_pp
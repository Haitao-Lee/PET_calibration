# coding = utf-8
import numpy as np
import os
import struct
from config import args
from tqdm import tqdm
import shutil

# --------------------------------------------
# 📁 Directory and File Management Functions
# --------------------------------------------

def count_top_level_folders(directory_path):
    """
    Count the number of top-level folders in a given directory.
    
    :param directory_path: Path to the directory.
    :return: Number of top-level folders.
    """
    folders = [folder for folder in os.listdir(directory_path) 
               if os.path.isdir(os.path.join(directory_path, folder))]
    return len(folders)

def get_files(path):
    """
    Recursively gather all file paths from a directory.
    
    :param path: Root directory to start searching.
    :return: List of full file paths.
    """
    names = []
    for root, _, files in os.walk(path):
        for file in files:
            names.append(os.path.join(root, file))
    return names

def copy_and_rename(src_file_path, dest_folder_path, new_file_name):
    """
    Copy a file to a new directory and rename it.
    
    :param src_file_path: Source file path.
    :param dest_folder_path: Destination folder path.
    :param new_file_name: New file name.
    """
    try:
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)
        dest_file_path = os.path.join(dest_folder_path, new_file_name)
        shutil.copy(src_file_path, dest_file_path)
    except IOError as e:
        print(f"File copy error: {e.strerror}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# --------------------------------------------
# 📊 Data Retrieval Functions
# --------------------------------------------

def get_data_file_names(data_dir, origin_lut, origin_peak):
    """
    Retrieve LUT and Peak file paths from a directory.
    
    :param data_dir: Root directory.
    :param origin_lut: LUT subfolder name.
    :param origin_peak: Peak subfolder name.
    :return: Tuple containing LUT, Peak_X, and Peak_Y file paths.
    """
    lut_names, peak_X_names, peak_Y_names = [], [], []
    for root, folders, _ in os.walk(data_dir):
        for i in folders:
            lut_folder = os.path.join(root, i, origin_lut)
            for _, _, files in os.walk(lut_folder):
                for k in range(len(files)):
                    lut_names.append(f"{lut_folder}/" + files[k])
            
            peak_folder = os.path.join(root, i, origin_peak)
            for _, _, files in os.walk(peak_folder):
                for k in range(len(files)):
                    if 'X Peak' in files[k]:
                        peak_X_names.append(f"{peak_folder}/" + files[k])
                        peak_Y_names.append(f"{peak_folder}/" + files[k].replace('X Peak', 'Y Peak'))
    return lut_names, peak_X_names, peak_Y_names

# --------------------------------------------
# 📈 Data Loading Functions
# --------------------------------------------

def load_origin_luts(names):
    """
    Load LUT files and convert them into NumPy arrays.
    
    :param names: List of LUT file paths.
    :return: NumPy array of LUTs.
    """
    luts = []
    for name in tqdm(names, desc="\033[31mLoading LUTs:\033[0m"):
        with open(name, 'rb') as raw_lut:
            size = os.path.getsize(name)
            tmp = [struct.unpack('>I', raw_lut.read(4))[0] for _ in range(size // 4)]
            lut = np.array(tmp).reshape([1024, 1024])
            luts.append(lut)
    return np.array(luts)

def load_origin_peaks(x_names, y_names):
    """
    Load Peak X and Peak Y files and convert them into NumPy arrays.
    
    :param x_names: List of Peak X file paths.
    :param y_names: List of Peak Y file paths.
    :return: Tuple containing Peak X and Peak Y NumPy arrays.
    """
    peaks_x, peaks_y = [], []
    for x_name, y_name in tqdm(zip(x_names, y_names), desc="\033[31mLoading Peaks:\033[0m"):
        with open(x_name, 'rb') as peak_x_f, open(y_name, 'rb') as peak_y_f:
            size_x, size_y = os.path.getsize(x_name), os.path.getsize(y_name)
            assert size_x == size_y, f'size_x: {size_x}; size_y: {size_y}'
            peak_xs = [struct.unpack('>I', peak_x_f.read(4))[0] for _ in range(size_x // 4)]
            peak_ys = [struct.unpack('>I', peak_y_f.read(4))[0] for _ in range(size_y // 4)]
        peaks_x.append(np.array(peak_xs).reshape([64, 64]))
        peaks_y.append(np.array(peak_ys).reshape([64, 64]))
    return np.array(peaks_x), np.array(peaks_y)

def load_training_data():
    """
    Load training LUT and Peak data from pre-processed files.
    :return: Tuple containing training LUTs and Peaks.
    """
    luts = [np.load(os.path.join(args.reg_luts_npy, 'training', path)) for path in os.listdir(args.reg_luts_npy + '/training/')]
    peaks = [np.load(os.path.join(args.reg_peaks_npy, 'training', path)) for path in os.listdir(args.reg_peaks_npy + '/training/')]
    return luts, peaks

def load_validation_data():
    """
    Load validation LUT and Peak data from pre-processed files.
    :return: Tuple containing validation LUTs and Peaks.
    """
    luts = [np.load(os.path.join(args.reg_luts_npy, 'validation', path)) for path in os.listdir(args.reg_luts_npy + '/validation/')]
    peaks = [np.load(os.path.join(args.reg_peaks_npy, 'validation', path)) for path in os.listdir(args.reg_peaks_npy + '/validation/')]
    return luts, peaks

# --------------------------------------------
# ✅ End of Script
# --------------------------------------------

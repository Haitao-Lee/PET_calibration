import numpy as np
import os
from tqdm import tqdm
import shutil
from config import args
import struct
import data_visualization
import copy
import matplotlib.pyplot as plt


def get_data_file_names(data_dir, origin_lut):
    """
    Retrieve file paths for LUT data from a given directory structure.
    
    This function traverses the specified root directory to locate files within
    the LUT subfolders. It collects all file paths for LUT data and returns them
    as a list.
    
    Args:
        data_dir (str): The root directory to search for LUT files.
        origin_lut (str): The name of the LUT subfolder.
    
    Returns:
        list: A list of absolute paths to LUT files.
    """
    lut_names = []
    
    # Traverse the root directory
    for root, folders, _ in os.walk(data_dir):
        for folder in folders:
            # Construct LUT folder path and collect all files
            lut_folder = os.path.join(root, folder, origin_lut)
            if os.path.exists(lut_folder):
                for file in os.listdir(lut_folder):
                    file_path = os.path.join(lut_folder, file)
                    if os.path.isfile(file_path):
                        lut_names.append(file_path)
    return lut_names


def read_raw_img_as_npy(path):
    """
    Read a raw image file and convert it into a NumPy array.
    
    This function reads a binary image file containing 32-bit unsigned integers 
    and reshapes the data into a 1024x1024 NumPy array.
    
    Args:
        path (str): Path to the raw image file.
    
    Returns:
        np.ndarray: A 1024x1024 NumPy array representing the image.
    """
    with open(path, 'rb') as raw_img:
        size = os.path.getsize(path)  # Get the file size in bytes
        img = [struct.unpack('>I', raw_img.read(4))[0] for _ in range(size // 4)]
        img = np.array(img).reshape([1024, 1024])
    return img


def read_raw_peak_as_npy(path):
    """
    Read a raw peak file and convert it into a NumPy array.
    
    This function reads a binary peak file containing 32-bit unsigned integers 
    and reshapes the data into a 64x64 NumPy array.
    
    Args:
        path (str): Path to the raw peak file.
    
    Returns:
        np.ndarray: A 64x64 NumPy array representing the peak data.
    """
    with open(path, 'rb') as peak_f:
        size_peak = os.path.getsize(path)  # Get the file size in bytes
        peak = [struct.unpack('>I', peak_f.read(4))[0] for _ in range(size_peak // 4)]
    return np.array(peak) # .reshape([64, 64])
            

def get_data_with_label(image, row_peak, col_peak):
    """
    Extract meaningful sub-images and their corresponding peak coordinates from a larger image.

    Parameters:
        image (numpy.ndarray): The input image array.
        row_peak (numpy.ndarray): Array of row indices (y-coordinates) for peak points.
        col_peak (numpy.ndarray): Array of column indices (x-coordinates) for peak points.

    Returns:
        tuple: 
            - effect_imgs (numpy.ndarray): Array of non-empty 256x256 sub-images.
            - effect_peaks (numpy.ndarray): Array of peak coordinates corresponding to each sub-image.
    """
    # Step 1: Create a binary image of the same shape as the input image
    #         where peak positions are marked with 1, and all other positions are 0.
    peaks_img = np.zeros(image.shape, dtype=np.uint8)
    row_peak = np.clip(row_peak, 0, image.shape[0]-1)
    col_peak = np.clip(col_peak, 0, image.shape[1]-1)
    # Step 2: Mark the peak positions on the binary image
    for i in range(row_peak.shape[0]):
        peaks_img[row_peak[i], col_peak[i]] = 1
    
    # Step 3: Prepare lists to store non-empty sub-images and their peak coordinates
    effect_imgs = []
    effect_peaks = []
    
    # Step 4: Divide the image into 16 (4x4) sub-images, each of size 256x256
    for i in range(4):  # Loop over rows of sub-images
        for j in range(4):  # Loop over columns of sub-images
            # Extract a sub-image and its corresponding binary peak region
            tmp_image = image[i*256:(i+1)*256, j*256:(j+1)*256]
            tmp_peaks_img = peaks_img[i*256:(i+1)*256, j*256:(j+1)*256]
            
            # Step 5: Check if the sub-image is non-empty (contains non-zero values)
            if tmp_image.max() != 0:
                effect_imgs.append(tmp_image)  # Store the valid sub-image
                effect_peaks.append(np.argwhere(tmp_peaks_img == 1))  # Store peak coordinates
    
    return effect_imgs, effect_peaks


def split_list_by_step(lst, step, start=0):
    """
    Extract elements from a list at regular intervals and return the selected 
    elements along with the remaining elements.

    Parameters:
        lst (list): The original list.
        step (int): The interval for selecting elements.
        start (int): The starting index for selection, default is 0.

    Returns:
        tuple: 
            - selected (list): A list containing elements selected at intervals.
            - remaining (list): A list containing all other elements not selected.
    """
    # Select every 'step' element from the list starting at index 'start'
    selected = lst[start::step]
    
    # Create a list of remaining elements that are not part of the selected indices
    remaining = [item for i, item in enumerate(lst) if i % step != start]
    
    return selected, remaining


def create_dataset(args):
    """
    Create training, validation, and test datasets from raw image and peak data.

    Parameters:
        args (object): An object containing dataset configurations, including:
            - raw_data (str): Root path for raw data.
            - raw_data_imgs (str): Sub-directory for raw images.
            - raw_data_peaks (str): Sub-directory for peak data.
            - img_name (str): File name template for images.
            - row_peak_name (str): File name template for row peaks.
            - col_peak_name (str): File name template for column peaks.
            - data_distribution (dict): Distribution rules for train, validation, and test sets.
            - valid_rate (float): Proportion of validation data.
            - train_img_dir (str): Directory to save training images.
            - train_label_dir (str): Directory to save training labels.
            - val_img_dir (str): Directory to save validation images.
            - val_label_dir (str): Directory to save validation labels.
            - test_img_dir (str): Directory to save test images.
            - test_label_dir (str): Directory to save test labels.

    Returns:
        None
    """
    # Step 1: Get all image file paths
    img_paths = get_data_file_names(args.raw_data, args.raw_data_imgs)
    
    # Initialize dataset containers
    train_val_set = []
    test_set = []
    
    # Step 2: Process each image and corresponding peak data
    for img_path in tqdm(img_paths, desc="\033[31mLoading data:\033[0m"):
        # Generate file paths for row and column peaks
        row_peak_path = img_path.replace(args.raw_data_imgs, args.raw_data_peaks).replace(args.img_name, args.row_peak_name)
        col_peak_path = row_peak_path.replace(args.row_peak_name, args.col_peak_name)
        
        # Load image and peak data
        img = read_raw_img_as_npy(img_path)
        row_peak = read_raw_peak_as_npy(row_peak_path)
        col_peak = read_raw_peak_as_npy(col_peak_path)
        
        # Get labeled data (sub-images and their peaks)
        imgs, peaks = get_data_with_label(img, row_peak, col_peak)
        
        # Determine if the image belongs to the test set
        test_sign = any(source in img_path for source in args.data_distribution['test_set'])
        
        # Distribute images and peaks into test or train_val sets
        target_set = test_set if test_sign else train_val_set
        target_set.extend(zip(imgs, peaks))
    
    # Step 3: Split the training and validation datasets
    valid_set, train_set = split_list_by_step(
        train_val_set, 
        int(1 / args.data_distribution['valid_rate'])
    )
    
    # Step 4: Save training data
    if not os.path.exists(args.train_img_dir):
        os.makedirs(args.train_img_dir)
    if not os.path.exists(args.train_label_dir):
        os.makedirs(args.train_label_dir)
    for i, (img, ps) in tqdm(enumerate(train_set), desc="\033[31mCreating Training Set:\033[0m"):
        np.save(os.path.join(args.train_img_dir, f'{i}.npy'), img)
        np.save(os.path.join(args.train_label_dir, f'{i}.npy'), ps)
    
    # Step 5: Save validation data
    if not os.path.exists(args.val_img_dir):
        os.makedirs(args.val_img_dir)
    if not os.path.exists(args.val_label_dir):
        os.makedirs(args.val_label_dir)
    for i, (img, ps) in tqdm(enumerate(valid_set), desc="\033[31mCreating Validation Set:\033[0m"):
        np.save(os.path.join(args.val_img_dir, f'{i}.npy'), img)
        np.save(os.path.join(args.val_label_dir, f'{i}.npy'), ps)
    
    # Step 6: Save test data
    if not os.path.exists(args.test_img_dir):
        os.makedirs(args.test_img_dir)
    if not os.path.exists(args.test_label_dir):
        os.makedirs(args.test_label_dir)
    for i, (img, ps) in tqdm(enumerate(test_set), desc="\033[31mCreating Test Set:\033[0m"):
        np.save(os.path.join(args.test_img_dir, f'{i}.npy'), img)
        np.save(os.path.join(args.test_label_dir, f'{i}.npy'), ps)


def process_data_in_folder(folder_path):
    """
    Display images one by one, allowing the user to delete, skip, or quit.

    Parameters:
        folder_path (str): Path to the folder containing image files.
    """
    # Get all .npy file paths in the folder
    peaks_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    if not peaks_files:
        print("No .npy files found in the folder.")
        return
    
    # Iterate through each image file
    for peaks_file in peaks_files:
        print(f"Processing: {peaks_file}")
        img_file = peaks_file.replace('label', 'img')
        
        # Load the image from the .npy file
        peaks = np.load(peaks_file)
        img = np.load(img_file)
        # img[img<np.mean(img)] = 0
        
        # Display the image using matplotlib
        fig, ax = plt.subplots()
        ax.matshow(img, alpha=1)
        ax.plot(peaks[:, 1], peaks[:, 0], "ro")
        ax.set_title(f"Image: {os.path.basename(peaks_file)}")
        plt.get_current_fig_manager().full_screen_toggle()        
        def on_key(event):
            """Handle key press events to delete, skip, or quit."""
            if event.key == '0':  # Delete the current file
                print(f"Deleting: {peaks_file}")
                plt.close(fig)  # Close the current figure
                os.remove(peaks_file)  # Remove the file from disk
            elif event.key == '1':  # Skip the current file
                print(f"Skipping: {peaks_file}")
                plt.close(fig)  # Close the current figure
            elif event.key == 'q':  # Quit the program
                print("Exiting program.")
                plt.close(fig)  # Close the current figure
                exit()
        
        # Bind the key press event to the handler
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

def data_augmentation(label_data_dir):
    """
    Perform data augmentation on image and label data.

    Augmentation includes:
        - Horizontal Flip
        - Vertical Flip
        - Rotation (90°, 180°, 270°)
        - Combined flips on rotated images

    Parameters:
        label_data_dir (str): Directory containing label (.npy) files.
    
    Returns:
        None
    """
    # Retrieve all .npy label files
    peaks_files = [os.path.join(label_data_dir, f) for f in os.listdir(label_data_dir) if f.endswith('.npy')]
    
    if not peaks_files:
        print("No .npy files found in the folder.")
        return
    
    # Iterate through each label file
    for peaks_file in tqdm(peaks_files, desc="\033[31mData Augmentation:\033[0m"):
        img_file = peaks_file.replace('label', 'img')
        
        # Load image and peak coordinates
        img = np.load(img_file)
        peaks = np.load(peaks_file)
        
        ## ---- Horizontal Flip ---- ##
        flip_lr_img = np.fliplr(img)
        np.save(img_file.replace('.npy', '_lr.npy'), flip_lr_img)
        peaks_lr = np.array([[row, 255 - col] for row, col in peaks])
        np.save(peaks_file.replace('.npy', '_lr.npy'), peaks_lr)
        
        ## ---- Vertical Flip ---- ##
        flip_ud_img = np.flipud(img)
        np.save(img_file.replace('.npy', '_ud.npy'), flip_ud_img)
        peaks_ud = np.array([[255 - row, col] for row, col in peaks])
        np.save(peaks_file.replace('.npy', '_ud.npy'), peaks_ud)
        
        ## ---- Rotation 90° Clockwise ---- ##
        rot_90_img = np.rot90(img, k=-1)
        np.save(img_file.replace('.npy', '_90.npy'), rot_90_img)
        peaks_90 = np.array([[col, 255 - row] for row, col in peaks])
        np.save(peaks_file.replace('.npy', '_90.npy'), peaks_90)
        
        ## ---- Horizontal Flip on 90° Rotated Image ---- ##
        rot_90_lr_img = np.fliplr(rot_90_img)
        np.save(img_file.replace('.npy', '_90_lr.npy'), rot_90_lr_img)
        peaks_90_lr = np.array([[col, row] for row, col in peaks])
        np.save(peaks_file.replace('.npy', '_90_lr.npy'), peaks_90_lr)
        
        ## ---- Vertical Flip on 90° Rotated Image ---- ##
        rot_90_ud_img = np.flipud(rot_90_img)
        np.save(img_file.replace('.npy', '_90_ud.npy'), rot_90_ud_img)
        peaks_90_ud = np.array([[255 - col, 255 - row] for row, col in peaks])
        np.save(peaks_file.replace('.npy', '_90_ud.npy'), peaks_90_ud)
        
        ## ---- Rotation 180° ---- ##
        rot_180_img = np.rot90(img, k=2)
        np.save(img_file.replace('.npy', '_180.npy'), rot_180_img)
        peaks_180 = np.array([[255 - row, 255 - col] for row, col in peaks])
        np.save(peaks_file.replace('.npy', '_180.npy'), peaks_180)
        
        ## ---- Rotation 270° Clockwise ---- ##
        rot_270_img = np.rot90(img, k=-3)
        np.save(img_file.replace('.npy', '_270.npy'), rot_270_img)
        peaks_270 = np.array([[255 - col, row] for row, col in peaks])
        np.save(peaks_file.replace('.npy', '_270.npy'), peaks_270)

        # rot_180_lr_img = np.fliplr(rot_180_img)
        # np.save(img_file.replace('.npy', '_180_lr.npy'), rot_180_lr_img)
        # peaks_180_lr = np.array([[255 - row, col] for row, col in peaks])
        # np.save(peaks_file.replace('.npy', '_180_lr.npy'), peaks_180_lr)
        # rot_180_ud_img = np.flipud(rot_180_img)
        # np.save(img_file.replace('.npy', '_180_ud.npy'), rot_180_ud_img)
        # peaks_180_ud = np.array([[row, 255 - col] for row, col in peaks])
        # np.save(peaks_file.replace('.npy', '_180_ud.npy'), peaks_180_lr)


def check_peaks(check_peaks, ref_peaks, rest_peaks, effect_margin):
    """
    Validate if peaks in `check_peaks` meet specified conditions based on `ref_peaks` and `rest_peaks`.

    Parameters:
        check_peaks (numpy.ndarray): An array of shape (n, 2) containing peaks to validate.
                                     Each row represents (row, col) coordinates.
        ref_peaks (numpy.ndarray): An array of shape (n, 2) containing reference peaks for validation.
                                   Each row represents (row, col) coordinates.
        rest_peaks (numpy.ndarray): An array of shape (m, 2) containing additional peaks for secondary validation.
                                    Each row represents (row, col) coordinates.
        effect_margin (float): Maximum allowable difference in column values between `check_peaks` 
                               and `ref_peaks` for valid peaks.

    Returns:
        bool: 
            - True if all peaks satisfy the conditions.
            - False if any peak fails to meet the criteria.

    Validation Criteria:
        1. For each peak in `rest_peaks`:
           - Its row value must not be less than the last row value of `check_peaks`.
           - Its column value must not exceed the last column value of `check_peaks`.
        2. For each peak in `check_peaks`:
           - Its row value must not be less than the corresponding row value in `ref_peaks`.
           - The absolute difference between column values of `check_peaks` and `ref_peaks` 
             must not exceed `effect_margin`.
    """
    if np.array(check_peaks).max() > 255:
        # data_visualization.image_label_visualization(np.zeros([256,256]), rest_peaks[:, 0], rest_peaks[:, 1], check_peaks)
        return False
    
    # Validate rest_peaks based on the last peak in check_peaks
    for i in range(rest_peaks.shape[0]):
        if rest_peaks[i, 0] < check_peaks[-1][0] and rest_peaks[i, 1] > check_peaks[-1][1]:
            # data_visualization.image_label_visualization(np.zeros([256,256]), rest_peaks[:, 0], rest_peaks[:, 1], check_peaks)
            return False
    
    # Validate check_peaks based on ref_peaks and effect_margin
    for i in range(len(check_peaks)):
        if check_peaks[i][0] < ref_peaks[i][0] or np.abs(check_peaks[i][1] - ref_peaks[i][1]) > effect_margin:
            # data_visualization.image_label_visualization(np.zeros([256,256]), rest_peaks[:, 0], rest_peaks[:, 1], check_peaks)
            return False

    # All conditions met
    return True


def top_n_values_with_indices(lst, n):
    """
    Find the top N largest values in a list along with their indices.

    Parameters:
        lst (list): The input list.
        n (int): The number of largest values to retrieve.

    Returns:
        list: A list of tuples, where each tuple contains:
              (value, index)
              representing the N largest values and their respective indices.
    """
    if len(lst) < n:
        raise ValueError("The list must contain at least N elements.")
    
    # Create a list of tuples (value, index)
    indexed_lst = list(enumerate(lst))
    
    # Sort the list by value in descending order
    sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)
    
    # Extract the top N largest values and their indices
    top_n = sorted_lst[:n]
    
    return [(value, index) for index, value in top_n]


def regular_peaks_Markov(peaks, peaks_num_per_row, thetas, active_space_size, distance_margin, effect_margin):
    """
    Regularize peak coordinates using a Markov process with angle constraints.
    
    Parameters:
        peaks (numpy.ndarray): Array of peak coordinates (n, 2).
        peaks_num_per_row (int): Number of peaks per row.
        thetas (list of float): List of angular constraints in degrees.
        active_space_size (int): Number of candidate points to consider.
        distance_margin (float): Maximum allowed distance for valid peaks.
        effect_margin (float): Maximum allowed column difference for valid peaks.
        
    Returns:
        tuple:
            - reg_peaks (numpy.ndarray): Regularized peak coordinates.
            - sign (bool): True if regularization succeeded, False otherwise.
    """
    peak_ = copy.deepcopy(peaks)
    reg_peaks = np.zeros_like(peaks)
    last_row_peaks = []
    cur_rows_peaks = [[] for _ in range(len(thetas))]
    peaks_cps = [copy.deepcopy(peak_) for _ in range(len(thetas))]

    for i in range(peaks.shape[0]):
        idxs = []
        for j, theta in enumerate(thetas):
            if i % peaks_num_per_row == 0:
                # First row: find the global minimum peak
                peaks_norm = np.sum(peaks_cps[j], axis=1)
                idx = np.argmin(peaks_norm).flatten()[0]
                idxs = [idx] * len(thetas)
            else:
                # Subsequent rows: calculate difference vectors and distances
                differ = peaks_cps[j] - np.expand_dims(cur_rows_peaks[j][-1], 0).repeat(peaks_cps[j].shape[0], axis=0)
                differ_norm = np.linalg.norm(differ, axis=1)
                sorted_indices = np.argsort(differ_norm)
                # Select the top 'active_space_size' elements from 'differ_norm' using 'sorted_indices' and add a small constant to avoid division by zero
                candidate_dist = differ_norm[sorted_indices[:active_space_size]] + 1e-5
                candidate_indices = sorted_indices[:active_space_size]
                
                indices, scores = [], []
                theta = np.deg2rad(theta)
                
                for k in range(active_space_size):
                    tmp_idx = candidate_indices[k]
                    tmp_differ = differ[tmp_idx, :]
                    indices.append(tmp_idx)
                    if candidate_dist[k] > distance_margin:
                        scores.append(0)
                    elif tmp_differ[1] / candidate_dist[k] > np.cos(theta) or (tmp_differ[0] < 0 and tmp_differ[1] > 0):
                        scores.append(np.max((tmp_differ[1] - tmp_differ[0] / np.tan(theta)) / (candidate_dist[k] ** 2), 0))
                    else:
                        scores.append(0)

                    if i // peaks_num_per_row != 0:
                        if last_row_peaks[i % peaks_num_per_row][0] > peaks_cps[j][tmp_idx, 0] or \
                                np.abs(last_row_peaks[i % peaks_num_per_row][1] - peaks_cps[j][tmp_idx][1]) > effect_margin:
                            scores[-1] = 0
                
                idx = indices[np.argmax(np.array(scores))]
                if np.array(scores).max() == 0:
                    a = 1
                # res = top_n_values_with_indices(scores, 2)
                # if candidate_dist[res[1][1]] * 2 < candidate_dist[res[0][1]]:
                #     idx = indices[res[1][1]]
                idxs.append(copy.deepcopy(idx))
            
            cur_rows_peaks[j].append(copy.deepcopy(peaks_cps[j][idxs[j], :]))
            peaks_cps[j][idxs[j], :] += 2 * peaks.shape[0]
        
        if len(cur_rows_peaks[0]) == peaks_num_per_row:
            sign = False
            for cur_row_peaks, peaks_cp in zip(cur_rows_peaks, peaks_cps):
                if i // peaks_num_per_row == 0:
                    last_row_peaks = copy.deepcopy(cur_row_peaks)
                sign = check_peaks(cur_row_peaks, last_row_peaks, peaks_cp, effect_margin)
                # data_visualization.image_label_visualization(np.zeros([256,256]), peaks_cp[:, 0], peaks_cp[:, 1], cur_row_peaks)
                if sign:
                    reg_peaks[((i // peaks_num_per_row) * peaks_num_per_row):((i // peaks_num_per_row) + 1) * peaks_num_per_row, :] = np.array(cur_row_peaks)
                    last_row_peaks = copy.deepcopy(cur_row_peaks)
                    peak_ = copy.deepcopy(peaks_cp)
                    peaks_cps = [copy.deepcopy(peak_) for _ in range(len(thetas))]
                    cur_rows_peaks = [[] for _ in range(len(thetas))]
                    break
            
            if not sign:
                for cur_row_peaks, peaks_cp in zip(cur_rows_peaks, peaks_cps):
                    if i // peaks_num_per_row == 0:
                        last_row_peaks = copy.deepcopy(cur_row_peaks)
                    # data_visualization.image_label_visualization(np.zeros([256, 256]), peaks[:, 0], peaks[:, 1], cur_row_peaks)
                return reg_peaks, False
    
    return reg_peaks, True


def label_regularization(label_data_dir, peaks_num_per_row, theta, active_space_size, distance_margin, effect_margin):
    """
    Apply regularization to peak coordinates in all .npy label files within a directory.
    
    Parameters:
        label_data_dir (str): Directory containing .npy label files.
        peaks_num_per_row (int): Number of peaks per row.
        theta (list of float): List of angular constraints in degrees.
        active_space_size (int): Number of candidate points to consider.
        distance_margin (float): Maximum allowed distance for valid peaks.
        effect_margin (float): Maximum allowed column difference for valid peaks.
    """
    peaks_files = [os.path.join(label_data_dir, f) for f in os.listdir(label_data_dir) if f.endswith('.npy')]
    success_num = 0
    
    for peaks_file in tqdm(peaks_files, desc="\033[31mLabel Regularization:\033[0m"):
        peaks = np.load(peaks_file)
        img = np.load(peaks_file.replace('label', 'img'))
        # data_visualization.image_label_visualization(img, peaks[:, 0], peaks[:, 1], peaks)

        reg_peaks, sign = regular_peaks_Markov(peaks, peaks_num_per_row, theta, active_space_size, distance_margin, effect_margin)
        # data_visualization.image_label_visualization(np.zeros([256,256]), peaks[:, 0], peaks[:, 1], reg_peaks)
        if sign:
            success_num += 1
            # np.save(peaks_file, reg_peaks)
        # else:
        #     os.remove(peaks_file) 
    print(f'Successfully regularized {success_num} labels, success rate: {success_num / len(peaks_files):.2f}')


def data_preprocess(args=args):
    # create_dataset(args=args)
    # process_data_in_folder(args.val_label_dir)
    # process_data_in_folder(args.train_label_dir)
    # process_data_in_folder(args.test_label_dir)
    # data_augmentation(args.train_label_dir)
    # data_augmentation(args.val_label_dir)
    # data_augmentation(args.test_label_dir)
    # label_regularization(args.train_label_dir, args.peaks_num_per_row, args.theta, args.markov_space_size, args.distance_margin, args.effect_margin)
    label_regularization(args.val_label_dir, args.peaks_num_per_row, args.theta, args.markov_space_size, args.distance_margin, args.effect_margin)
    label_regularization(args.test_label_dir, args.peaks_num_per_row, args.theta, args.markov_space_size, args.distance_margin, args.effect_margin)
    
    pass


if __name__ == '__main__':
    data_preprocess(args=args)
    pass
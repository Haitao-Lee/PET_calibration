import argparse
import os
import torch
import torch.nn as nn
import Loss

# -----------------------------
# Argument Parser Configuration
# -----------------------------

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

# Initialize argument parser
parser = argparse.ArgumentParser(description='Configuration parser for deep learning model training.')

# -----------------------------
# Model Saving Parameters
# -----------------------------
parser.add_argument('--save_model', default='./checkpoints',
                    help='Path to save trained model checkpoints.')

# -----------------------------
# Data Parameters
# -----------------------------
parser.add_argument('--raw_data', default='./raw_data',
                    help='Root directory containing raw data.')
parser.add_argument('--raw_data_imgs', default='EventCounts2DMap',
                    help='Sub-directory for raw image data.')
parser.add_argument('--raw_data_peaks', default='EventCountsPeak',
                    help='Sub-directory for raw peak data.')
parser.add_argument('--img_name', default='lut',
                    help='Identifier for LUT image data.')
parser.add_argument('--row_peak_name', default='X Peak',
                    help='Identifier for X-axis peak data.')
parser.add_argument('--col_peak_name', default='Y Peak',
                    help='Identifier for Y-axis peak data.')
parser.add_argument('--data_distribution',
                    default={
                        'train_val_set': ['Hospital1', 'Hospital2', 'Hospital3', 'Hospital4'],
                        'test_set': ['Hospital5'],
                        'valid_rate': 0.2,
                    },
                    help='Configuration for dataset splitting into training, validation, and testing.')
parser.add_argument('--peaks_num_per_row', default=16, type=int,
                    help='Number of peaks per row.')
parser.add_argument('--markov_space_size', default=8, type=int,
                    help='Size of the Markov search space.')
parser.add_argument('--theta', default=[60, 45, 30], type=list,
                    help='List of angular configurations for processing.')
parser.add_argument('--distance_margin', default=50, type=int,
                    help='Allowed margin for distance calculations.')
parser.add_argument('--effect_margin', default=10, type=int,
                    help='Allowed margin for effect-based calculations.')

# -----------------------------
# Training Parameters
# -----------------------------
parser.add_argument('--epochs', default=200, type=int,
                    help='Total number of training epochs.')
parser.add_argument('--patience', default=100, type=int,
                    help='Number of epochs to wait for improvement before early stopping.')
parser.add_argument('--delta', default=1e-6, type=float,
                    help='Minimum change in monitored value for early stopping.')
parser.add_argument('--train_loss', default= nn.MSELoss(),
                    help='Loss function used during training.')
parser.add_argument('--valid_loss', default= nn.MSELoss(), # Loss.nearest_distance_loss, # 
                    help='Loss function used during validation.')
parser.add_argument('--lr', default=5e-4, type=float,
                    help='Learning rate for the optimizer.')
parser.add_argument('--loss_eps', default=30, type=int,
                    help='Threshold value for acceptable loss.')

# -----------------------------
# Dataset Paths
# -----------------------------
parser.add_argument('--train_img_dir', default=os.path.join(current_dir, 'dataset/train/img'),
                    help='Directory containing training images.')
parser.add_argument('--train_label_dir', default=os.path.join(current_dir, 'dataset/train/label'),
                    help='Directory containing training labels.')
parser.add_argument('--val_img_dir', default=os.path.join(current_dir, 'dataset/val/img'),
                    help='Directory containing validation images.')
parser.add_argument('--val_label_dir', default=os.path.join(current_dir, 'dataset/val/label'),
                    help='Directory containing validation labels.')
parser.add_argument('--test_img_dir', default=os.path.join(current_dir, 'dataset/test/img'),
                    help='Directory containing test images.')
parser.add_argument('--test_label_dir', default=os.path.join(current_dir, 'dataset/test/label'),
                    help='Directory containing test labels.')
parser.add_argument('--output', default=os.path.join(current_dir, 'output'),
                    help='Directory to save model predictions.')

# -----------------------------
# Experiments
# -----------------------------
parser.add_argument('--exp_output', default=os.path.join(current_dir, 'output'),
                    help='Directory containing experiments results.')

# -----------------------------
# Hardware Settings
# -----------------------------
parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    help='Device to run the training process (cuda or cpu).')

# -----------------------------
# Parse Arguments
# -----------------------------
args = parser.parse_args()

# -----------------------------
# Display Configuration
# -----------------------------
print("\n--- Configuration Parameters ---")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
print("---------------------------------")

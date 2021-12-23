# Libs
import os
import sys

# Own modules
import preprocess
import prepare_input
import train_variants
import progress
import argparse

parser = argparse.ArgumentParser(description='Train / test')
parser.add_argument('--method',default = 0, type=int)
parser.add_argument('--type',default ='all', type=str)

args = parser.parse_args()

# Constants
SIZE = 512

# Helper functions
def relPath(dir):
    "Returns path of directory relative to the executable"
    return os.path.join(os.path.dirname(__file__), dir)

# Crop and resize images
# This expects the images to be saved in the data folder
# Extract 1/4 more for cropping augmentation
print('Preprocessing...')
# input_dir = '/media/data/mu/ML2/data2/our_data/Diabetes'
# output_dir = '/media/data/mu/ML2/data2/our_data_processed'
# preprocess.preprocess(input_dir, output_dir, size=int(SIZE*1.1))
preprocess.preprocess(relPath('data'), relPath('preprocessed'), size=int(SIZE*1.1))

# Prepare input: convert to float with unit variance and zero mean,
# extract labels and save everything as a big numpy array to be used for training
print('Preparing input...')
prepare_input.prepare(relPath('preprocessed'), relPath('input'))

# print command to start tensorboard
progress.start_tensorboard()

# Train network
if '--cross-validation' in sys.argv:
    train_variants.train_cross_validation(relPath('input'), sets=3, size=SIZE)
else:
    train_variants.train_single(relPath('input'), size=SIZE, method=args.method, type=args.type)  # method=( original/ ori_train+our_test / our_train/test)

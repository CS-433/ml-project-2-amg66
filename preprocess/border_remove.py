import os
import sys
import numpy as np
import imageio
import skimage.transform
import glob

def remove_border(img, threshold=0):
    "Crop image, throwing away the border below the threshold"
    mask = img > threshold
    return img[np.ix_(mask.any(1), mask.any(0))]

def crop_center(img, size):
    "Crop center sizexsize of the image"
    y, x = img.shape
    startx = (x - size) // 2 
    starty = (y - size) // 2
    return img[starty:starty+size, startx:startx+size]

def bigger_edge(img):
    y, x = img.shape
    return y if y < x else x

def preprocess(inDir, outDir, size=512):
    "Preprocess files, resizing them to sizexsize pixels and removing black borders"

    # Ensure output folder exists
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    files = glob.glob(f'{inDir}/*/*/*.*')
    num = len(files)

    print('files', files)

    for i, path in enumerate(files):

        prefix = os.path.split(path)[0][len(inDir)+1:]
        print('dir', inDir, prefix)
        os.path.split(path)[0]

        f = os.path.split(path)[1]
        in_path = path
        out_dir = os.path.join(outDir, prefix)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, f)

        if os.path.exists(out_path):
            # If the file was already preprocessed, do nothing
            continue

        
        print('Preprocessing {} - {} %'.format(f, int(i / num * 100)), end='\r')

        print('inpath', in_path)

        img = imageio.imread(in_path)

        # If the image is RGB, compress it
        if len(img.shape) > 2:
            img = img.mean(2)

        # PREPROCESSING
        # Remove black border (sometimes there is a black band)
        img_noborder = remove_border(img)
        # Find bigger edge
        edge = bigger_edge(img_noborder)
        # Crop center
        img_cropped = crop_center(img_noborder, edge)
        # Resize to final size
        img_resized = skimage.transform.resize(img_cropped, (size, size), order=3)

        imageio.imsave(out_path, img_resized)
        print('save to ', out_path)

print('Preprocessing...')
input_dir = '/media/data/mu/ML2/data2/our_data'
output_dir = '/media/data/mu/ML2/data2/our_data_processed'
SIZE = 512
preprocess(input_dir, output_dir, size=int(SIZE*1.1))
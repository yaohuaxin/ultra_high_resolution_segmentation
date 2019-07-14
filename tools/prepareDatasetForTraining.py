#
# Before run this script, you need to unzip the datasets in sub-folds.
# If you not want to one dataset not been traned, please not unzip it.
#
import os
from glob import glob
from PIL import Image
from skimage import io
from skimage import transform
from skimage.util import img_as_ubyte
import openslide
import numpy as np
import argparse

import multiprocessing
from multiprocessing import Pool

root_path = "/home/shhxyao/huaxin/projects/ai/contest/DatasetPAIP2019/"

parser = argparse.ArgumentParser(description='Prepare PAIP 2019 dataset for traning.')
parser.add_argument('--data_path', type=str, default=root_path, help='path to dataset where images store')
args = parser.parse_args()

def resize(packageFold):
    print(multiprocessing.current_process())
    print("==== ==== Enter fold: ", packageFold)
    
    #
    # Read SVS
    #
    wsi_path = glob(packageFold + '/*.[S, s][V, v][S, s]')[0]
    #print("wsi_path        : ", wsi_path)
    wsi = openslide.OpenSlide(wsi_path)
    print("WSI Leveles: ", wsi.level_count, "; dimensions (width, height): ", wsi.level_dimensions)
    '''
    img_0 = wsi.read_region((0, 0), 0, wsi.level_dimensions[0]).convert('RGB')
    print("Image level 0 size (width, height)", img_0.size)
    '''
    img_1 = wsi.read_region((0, 0), 1, wsi.level_dimensions[1]).convert('RGB')
    print("Image level 1 size (width, height)", img_1.size)

    img_2 = wsi.read_region((0, 0), 2, wsi.level_dimensions[2]).convert('RGB')
    print("Image level 2 size (width, height)", img_2.size)
    
    #
    # Read viable mask
    #
    viable_mask_path = glob(packageFold+"/*viable.tif")[0]
    #print("viable_mask_path: ", viable_mask_path)
    viable_mask = io.imread(viable_mask_path)
    print("viable_mask shap (rows, columns): ", viable_mask.shape)
    print("viable_mask data type: ", viable_mask.dtype)
    #viable_mask = img_as_ubyte(viable_mask)
    viable_mask = viable_mask.astype(np.float)
    #print("viable_mask data type: ", viable_mask.dtype)
    #io.imshow(viable_mask)
    #io.show()
    
    #print("Resize from:", viable_mask.shape, " to: ", (viable_mask.shape[0]//2, viable_mask.shape[1]//2))
    #viable_mask_resize = transform.resize(viable_mask, (viable_mask.shape[0]//2, viable_mask.shape[1]//2))
    #io.imshow(viable_mask_resize)
    #io.show()
    
    print("Resize viable_mask, from: ", viable_mask_path)
    print("                      to: ", viable_mask_path.replace("viable.tif", "viable_level_1.tif"))
    viable_mask_level_1 = transform.resize(viable_mask, (img_1.size[1], img_1.size[0]))
    print("viable_mask_level_1 shap (rows, columns): ", viable_mask_level_1.shape)
    io.imsave(viable_mask_path.replace("viable.tif", "viable_level_1.tif"), viable_mask_level_1)
    
    print("Resize viable_mask, from: ", viable_mask_path)
    print("                      to: ", viable_mask_path.replace("viable.tif", "viable_level_2.tif"))
    viable_mask_level_2 = transform.resize(viable_mask, (img_2.size[1], img_2.size[0]))
    print("viable_mask_level_2 shap (rows, columns): ", viable_mask_level_2.shape)
    io.imsave(viable_mask_path.replace("viable.tif", "viable_level_2.tif"), viable_mask_level_2)
    #io.imshow(viable_mask_level_2)
    #io.show()
    #print(viable_mask_level_2)
    
    #
    # Read whole mask
    #
    whole_mask_path = glob(packageFold+"/*whole.tif")[0]
    #print("whole_mask_path : ", whole_mask_path)
    whole_mask = io.imread(whole_mask_path)
    print("whole_mask shap (rows, columns): ", whole_mask.shape)
    print("whole_mask data type: ", whole_mask.dtype)
    whole_mask = whole_mask.astype(np.float)
    
    print("Resize whole_mask, from: ", whole_mask_path)
    print("                     to: ", whole_mask_path.replace("whole.tif", "whole_level_1.tif"))
    whole_mask_level_1 = transform.resize(whole_mask, (img_1.size[1], img_1.size[0]))
    print("whole_mask_level_1 shap (rows, columns): ", whole_mask_level_1.shape)
    io.imsave(whole_mask_path.replace("whole.tif", "whole_level_1.tif"), whole_mask_level_1)
    
    print("Resize whole_mask, from: ", whole_mask_path)
    print("                     to: ", whole_mask_path.replace("whole.tif", "whole_level_2.tif"))
    whole_mask_level_2 = transform.resize(whole_mask, (img_2.size[1], img_2.size[0]))
    print("whole_mask_level_2 shap (rows, columns): ", whole_mask_level_2.shape)
    io.imsave(whole_mask_path.replace("whole.tif", "whole_level_2.tif"), whole_mask_level_2)

    print("==== ==== Leave fold: ", packageFold)


def main():
    # PAIP 2019 dataset is divided into 2 phases, and stored in two different folds:
    # Training_phase_1, and Training_phase_2
    # Each phase, contains several datasets
    # Each dataset, contains a WSI (.svs), and two mask files (.tif)
    phases = [x for x in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, x))]
    
    packageFolds = []
    for phase in phases:
        phaseFold = os.path.join(args.data_path, phase)
        packages = [x for x in os.listdir(phaseFold) if os.path.isdir(os.path.join(phaseFold, x))]
        
        for package in packages:
            packageFolds.append(os.path.join(phaseFold, package))
            
    for fold in packageFolds:
        print(fold)
    
    p = Pool(6)
    p.map(resize, packageFolds)

    

if __name__ == "__main__":
    main()


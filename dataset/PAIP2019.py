from dataset.deep_globe import is_image_file

import os
# from os.path import join
import torch.utils.data as data
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import skimage
import skimage.io as skimage_io
import openslide
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

import dataset.deep_globe

class PAIP2019(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, ids, label=False, transform=False, image_level=2):
        super(PAIP2019, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.image_level = image_level
        self.classdict = {1: "normal", 2: "abnormal", 0: "unknown"}
        
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)
        self.resizer = transforms.Resize((2448, 2448))

    def __getitem__(self, index):
        #print(self.ids[index])
        image_and_labels = [image_file for image_file in os.listdir(os.path.join(self.root, self.ids[index])) if is_image_file(image_file)]
        #print(image_and_labels)
        
        for item in image_and_labels:
            if item.endswith("svs") or item.endswith("SVS"):
                image_file = item
                #print("image_file: ", image_file)
                
            if item.endswith("viable_level_1.tif"):
                local_mask_file_level_1 = item
                #print("local_mask_file: ", local_mask_file)
            
            if item.endswith("whole_level_1.tif"):
                global_mask_file_level_1 = item
                #print("global_mask_file: ", global_mask_file)
            
            if item.endswith("viable_level_2.tif"):
                local_mask_file_level_2 = item
                #print("local_mask_file: ", local_mask_file)
            
            if item.endswith("whole_level_2.tif"):
                global_mask_file_level_2 = item
                #print("global_mask_file: ", global_mask_file)


        sample = {}
        sample['id'] = self.ids[index]

        #print("Open WSI file: ", os.path.join(self.root, self.ids[index], image_file))
        wsi = openslide.OpenSlide(os.path.join(self.root, self.ids[index], image_file))
        image = wsi.read_region((0, 0), self.image_level, wsi.level_dimensions[self.image_level]).convert('RGB')
        sample['image'] = image
        #print(type(sample['image']))
        #print("image.size (width, height): ", image.size)
        #skimage.io.imshow(np.array(sample['image']))
        #skimage.io.show()

        # sample['image'] = transforms.functional.adjust_contrast(image, 1.4)
        #print("Open global mask file: ", os.path.join(self.root, self.ids[index], global_mask_file))
        if self.label:
            # label = scipy.io.loadmat(join(self.root, 'Notification/' + self.ids[index].replace('_sat.jpg', '_mask.mat')))["label"]
            # label = Image.fromarray(label)            
            # label = Image.open(os.path.join(self.root, self.ids[index]))
            if self.image_level == 0:
                pass
            elif self.image_level == 1:
                label = skimage_io.imread(os.path.join(self.root, self.ids[index], global_mask_file_level_1))
            elif self.image_level == 2:
                label = skimage_io.imread(os.path.join(self.root, self.ids[index], global_mask_file_level_2))
            else:
                pass
            #print(type(label))
            #print("label.shap (rows, columns): ", label.shape)
            
            label = Image.fromarray(label)
            #print(type(label))
            #print("label.size (width, height): ", label.size)
            
            sample['label'] = label
            
            #skimage.io.imshow(np.array(sample['label']))
            #skimage.io.show()

        if self.transform and self.label:
            #print("Performance transform: ")
            image, label = self._transform(image, label)
            sample['image'] = image
            sample['label'] = label
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}
        return sample

    def _transform(self, image, label):
        # if np.random.random() > 0.5:
        #     image = self.color_jitter(image)

        # if np.random.random() > 0.5:
        #     image = transforms.functional.vflip(image)
        #     label = transforms.functional.vflip(label)

        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        if np.random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, degree)
            label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     degree = 60 * np.random.random() - 30
        #     image = transforms.functional.rotate(image, degree)
        #     label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     ratio = np.random.random()
        #     h = int(2448 * (ratio + 2) / 3.)
        #     w = int(2448 * (ratio + 2) / 3.)
        #     i = int(np.floor(np.random.random() * (2448 - h)))
        #     j = int(np.floor(np.random.random() * (2448 - w)))
        #     image = self.resizer(transforms.functional.crop(image, i, j, h, w))
        #     label = self.resizer(transforms.functional.crop(label, i, j, h, w))
        
        return image, label


    def __len__(self):
        return len(self.ids)
"""
Class for CubiCasa5k dataset. 
@author: r.kippers, 2021
"""

from collections import defaultdict 
import itertools
import os
from readers.svg_reader import CubiCasaSvgReader
import tensorflow as tf
import numpy as np

class CubiCasaDatset():

    def __init__(self,split=0.2,preserve_aspect_ratio=False):
        """
        Dataset class for semantic segmentation 

        Parameters 
        ----------
        split : float
            Split ratio train/test data 
        preserve_aspect_ratio : boolean
            Preserve aspect ratio
        """

        # Dictionary for image and mask paths
        self.data = defaultdict()
        self.image_size = 512
        self.split = split 
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        image_folder_paths = [list(map(lambda x : 'input/cubicasa5k/' + folder + "/" + x, os.listdir("input/cubicasa5k/" + folder))) for folder in ["colorful", "high_quality", "high_quality_architectural"]]
        image_folder_paths = list(itertools.chain(*image_folder_paths))

        for i,image_folder_path in enumerate(image_folder_paths):
            # Set folder paths
            image_path = os.path.join(image_folder_path, "F1_scaled.png")
            mask_path = os.path.join(image_folder_path, "model.svg")

            self.data[i] = {
                "folder_path": image_folder_path,
                "image_path": image_path,
                "mask_path": mask_path
            }

    def get_sample(self, index):
        """
        Returns sample 

        Parameters
        ----------
        index : int
            Index of sample 

        Output
        ------
        object {folder_path (string), image (tensor), mask (tensor)}
        """
        image, img_shape = self.load_image(self.data[index]["image_path"])
        mask = self.load_mask(self.data[index]["mask_path"],img_shape)

        return {
            "folder_path": self.data[index]["folder_path"],
            "image":image,
            "mask":mask
        }


    def load_image(self,image_path):
        """
        Load image as Tensor

        Parameters
        ----------
        image_path : string
            Path to image 
        
        Output
        ------
        tf.image 
        tuple 
            Original image shape (x,y)
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image,channels=1) # output grayscale 
        image_shape = image.shape[0:2]
        #image = tf.image.convert_image_dtype(image, tf.float32)
        if self.preserve_aspect_ratio: 
            image = tf.image.resize_with_pad(image, self.image_size, self.image_size)
        else: 
            image = tf.image.resize(image, [self.image_size, self.image_size])
        return image, image_shape

    def load_mask(self, mask_path, image_original_shape):
        """
        Load Mask as Tensor

        Parameters
        ----------
        mask_path : string 
            Path to mask 
        image_original_shape : tuple 
            Original image shape (x,y)

        Output
        ------
        tf.image 
        """

        svg_parser = CubiCasaSvgReader(mask_path,image_original_shape)
        svg_parser.read()
        mask = tf.convert_to_tensor(svg_parser.get_walls())
        mask = tf.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        if self.preserve_aspect_ratio:
            mask = tf.image.resize_with_pad(mask, image_original_shape[0], image_original_shape[1])
        else: 
            mask = tf.image.resize(mask, (image_original_shape[0], image_original_shape[1]))

        mask = tf.image.resize(mask, [self.image_size, self.image_size])
        return mask

    def get_tf_dataset(self, split_set,batch=8):
        """
        Returns tf dataset 

        Parameters
        ----------
        split_set : string
        training or validation 
        """

        num_samples = len(self.data.keys())

        if split_set == "training":
            indices = [i for i in range(0, np.floor(num_samples * (1- self.split)).astype('int'))]
        else: 
            indices = [i for i in range(np.ceil(num_samples * (1- self.split)).astype('int'), num_samples)]

        dataset = tf.data.Dataset.from_tensor_slices((indices))
        dataset = dataset.map(self.tf_parse) 
        dataset = dataset.batch(batch)
        dataset = dataset.repeat() 
        return dataset 


    def _parse(self,i):
        """Todo write docs"""
        sample = self.get_sample(i)
        return sample["image"], sample["mask"]

    def tf_parse(self,i):
        """Todo write docs"""
        x,y = tf.numpy_function(self._parse, [i] , [tf.float32, tf.float32])
        x.set_shape([self.image_size, self.image_size, 1])
        y.set_shape([self.image_size, self.image_size, 1])
        return x,y


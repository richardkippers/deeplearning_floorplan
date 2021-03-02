"""
Tensorflow Class for CubiCasa5k dataset. 
@author: r.kippers, 2021
"""

from collections import defaultdict 
import itertools
import os
from readers.svg_reader import CubiCasaSvgReader
import tensorflow as tf
import numpy as np

from dataset.bbox_utils import swap_xy, convert_to_xywh, convert_to_corners, compute_iou

cubicasa_openings_dict = { 
    "window":0,
    "door":1
}

class CubiCasaDatset():

    def __init__(self,split=0.2,preserve_aspect_ratio=False):
        """
        Dataset class for CubiCasa5K for semantic segmentation and 
        object detection in TF 2 / Keras
        
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
            vector_path = os.path.join(image_folder_path, "model.svg")

            self.data[i] = {
                "folder_path": image_folder_path,
                "image_path": image_path,
                "vector_path": vector_path
            }

    def get_sample(self, index, include_wall_mask=False, include_openings=False):
        """
        Returns sample 

        Parameters
        ----------
        index : int
            Index of sample 
        include_wall_mask : boolean
            Include Wall Mask 
        include_openings : boolean
            Include openings bounding box
        
        Output
        ------
        object {
            folder_path (string), image (tensor), wall_mask (tensor), openings_class (tensor) openings_bbox (tensor)
        }
        """

        image, img_orig_shape = self.load_image(self.data[index]["image_path"])

        wall_mask, openings_y, openings_bbox = None, None, None

        if include_wall_mask:
            wall_mask = self.load_mask(self.data[index]["vector_path"],img_orig_shape,"WALL")

        if include_openings: 
            openings_y, openings_bbox = self.load_openings(self.data[index]["vector_path"], img_orig_shape)

        return {
            "folder_path": self.data[index]["folder_path"],
            "image":image,
            "wall_mask":wall_mask,
            "openings_class":openings_y, 
            "openings_bbox": openings_bbox
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
        # if self.preserve_aspect_ratio: 
        #     image = tf.image.resize_with_pad(image, self.image_size, self.image_size)
        # else: 
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image, image_shape

    def load_mask(self, vector_path, image_original_shape, mask_type):
        """
        Load Wall Mask as Tensor

        Parameters
        ----------
        vector_path : string 
            Path to mask 
        image_original_shape : tuple 
            Original image shape (x,y)
        mask_type : string 
            Object type, only WALL is supported
        Output
        ------
        tf.image 
        """
        svg_parser = CubiCasaSvgReader(vector_path,image_original_shape)
        svg_parser.read()

        if mask_type == "WALL": 
            mask = svg_parser.get_walls()

        mask = tf.convert_to_tensor(mask)
        mask = tf.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        # if self.preserve_aspect_ratio:
        #     #TODO this scales the mask to top. Don't use this yet! 
        #     mask = tf.image.resize_with_pad(mask, image_original_shape[0], image_original_shape[1])
        # else: 
        mask = tf.image.resize(mask, (image_original_shape[0], image_original_shape[1]))

        mask = tf.image.resize(mask, [self.image_size, self.image_size])
        return mask


    def load_openings(self, vector_path, image_original_shape): 
        """
        Load and scale openings as tensor 
        
        Parameters
        ----------
        vector_path : string
            Path to vector
        image_original_shape : tuple 
            Original image shape (x,y)
        
        Output
        ------
        y : tf.Tensor 
            Labels in 1d 
        bboxes : tf.Tensor 
            bbox 
        """
        
        scale_factor_x =  self.image_size / image_original_shape[1] 
        scale_factor_y =  self.image_size / image_original_shape[0] 
        
        svg_parser = CubiCasaSvgReader(vector_path,image_original_shape)
        svg_parser.read()
        
        openings = svg_parser.get_openings() 
        openings = np.array(openings,dtype=object)

        y = tf.cast(np.vectorize(cubicasa_openings_dict.get)(openings[:,0]),tf.int32)
        
        bboxes = np.zeros((0,4))
        for i in range(len(openings)):
            bbox = openings[i,1]
            bbox = np.array(bbox)
            bbox[0] = bbox[0] * scale_factor_x
            bbox[1] = bbox[1] * scale_factor_y
            bbox[2] = bbox[2] * scale_factor_x
            bbox[3] = bbox[3] * scale_factor_y
            bboxes = np.vstack([bboxes, bbox])

        return y, bboxes
        


    def get_tf_dataset_wall_mask(self, split_set, batch=8):
        """
        Returns TensorFlow dataset for wall mask

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
        dataset = dataset.map(self.tf_parse_wall_mask) 
        dataset = dataset.batch(batch)
        dataset = dataset.repeat() 
        return dataset 


    def _tf_do_get_wall_mask(self,i):
        """Todo write docs"""
        sample = self.get_sample(i)
        return sample["image"], sample["wall_mask"]

    def tf_parse_wall_mask(self,i):
        """Todo write docs"""
        x,y = tf.numpy_function(self._tf_do_get_wall_mask, [i] , [tf.float32, tf.float32])
        x.set_shape([self.image_size, self.image_size, 1])
        y.set_shape([self.image_size, self.image_size, 1])
        return x,y


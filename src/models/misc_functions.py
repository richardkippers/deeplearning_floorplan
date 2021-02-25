"""
Misc. functions for ML 
@author: r.kippers, 2021
"""

import numpy as np
import tensorflow as tf

def iou(y_true, y_pred):
    """
    The iou function calculate the Intersection Over Union (IOU) between 
    the ground truth (y_true) and the predicted output (y_pred)
    """
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)
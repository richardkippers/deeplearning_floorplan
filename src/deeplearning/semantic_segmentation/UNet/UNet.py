"""
UNet implementation for Tensorflow
Source: https://idiotdeveloper.com/polyp-segmentation-using-unet-in-tensorflow-2/
@author: r.kippers, 2021 
"""

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class UNetModelBuilder():
    """
    TODO improve and extend default TF class 
    """

    def __init__(self,low_gpu_memory):
        """
        TODO improve and extend default TensorFlow class

        Parameters
        ----------
        low_gpu_memory : boolean 
            Decrease model size if true 
        """

        self.low_gpu_memory = low_gpu_memory

    def conv_block(self, x, num_filters):
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def build_model(self,image_size=512):
        
        num_filters = [16, 32, 48, 64] # 414,113 params
        
        if self.low_gpu_memory:
            num_filters = [2]  # 297 params
        
        inputs = Input((image_size, image_size, 1))

        skip_x = []
        x = inputs

        ## Encoder
        for f in num_filters:
            x = self.conv_block(x, f)
            skip_x.append(x)
            x = MaxPool2D((2, 2))(x)

        ## Bridge
        x = self.conv_block(x, num_filters[-1])

        num_filters.reverse()
        skip_x.reverse()

        ## Decoder
        for i, f in enumerate(num_filters):
            x = UpSampling2D((2, 2))(x)
            xs = skip_x[i]
            x = Concatenate()([x, xs])
            x = self.conv_block(x, f)

        ## Output
        x = Conv2D(1, (1, 1), padding="same")(x)
        x = Activation("sigmoid")(x)

        return Model(inputs, x)
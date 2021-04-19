"""
UNet implementation for Tensorflow
Source: https://medium.com/deep-learning-journals/fast-scnn-explained-and-implemented-using-tensorflow-2-0-6bd17c17a49e
@author: r.kippers, 2021 
"""


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class Fast_SCNNBuilder():

    def conv_block(self,inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
        if(conv_type == 'ds'):
            x = SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
        else:
            x = Conv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)  

        x = BatchNormalization()(x)

        if (relu):
            x = tf.keras.activations.relu(x)
        
        return x

    def _res_bottleneck(self,inputs, filters, kernel, t, s, r=False):
    
        tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

        x = self.conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        x = self.conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

        if r:
            x = add([x, inputs])
        return x

    def bottleneck_block(self,inputs, filters, kernel, t, strides, n):
        x = self._res_bottleneck(inputs, filters, kernel, t, strides)
        
        for i in range(1, n):
            x = self._res_bottleneck(x, filters, kernel, t, 1, True)

        return x

    def pyramid_pooling_block(self, input_tensor, bin_sizes):

        concat_list = [input_tensor]
        w = 16
        h = 16

        for bin_size in bin_sizes:
            x = AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
            x = Conv2D(128, 3, 2, padding='same')(x)
            x = Lambda(lambda x: tf.image.resize(x, (w,h)))(x)

            concat_list.append(x)

        return concatenate(concat_list)

    def build_model(self,image_size=512):
        
        """
        Build Fast-SCNN Network
        todo rm tf.keras.layers
        """
        
        input_layer = Input(shape=(512, 512, 3), name = 'input_layer')

        lds_layer = self.conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))
        lds_layer = self.conv_block(lds_layer, 'ds', 48, (3, 3), strides = (2, 2))
        lds_layer = self.conv_block(lds_layer, 'ds', 64, (3, 3), strides = (2, 2))

        gfe_layer = self.bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
        gfe_layer = self.bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
        gfe_layer = self.bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)

        gfe_layer = self.pyramid_pooling_block(gfe_layer, [2,4,6,8])

        ff_layer1 = self.conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False)

        ff_layer2 = UpSampling2D((4, 4))(gfe_layer)
        ff_layer2 = DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
        ff_layer2 = BatchNormalization()(ff_layer2)
        ff_layer2 = tf.keras.activations.relu(ff_layer2)
        ff_layer2 = Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)

        ff_final = add([ff_layer1, ff_layer2])
        ff_final = BatchNormalization()(ff_final)
        ff_final = tf.keras.activations.relu(ff_final)

        classifier = SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
        classifier = BatchNormalization()(classifier)
        classifier = tf.keras.activations.relu(classifier)

        classifier = SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv2_classifier')(classifier)
        classifier = BatchNormalization()(classifier)
        classifier = tf.keras.activations.relu(classifier)

        classifier = self.conv_block(classifier, 'conv', 1, (1, 1), strides=(1, 1), padding='same', relu=True) # modified 19 to 1

        classifier = Dropout(0.3)(classifier)

        classifier = UpSampling2D((8, 8))(classifier)
        classifier = tf.keras.activations.softmax(classifier)

        model = Model(inputs= input_layer , outputs= classifier, name = 'Fast_SCNN')

        return model

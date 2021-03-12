import matplotlib.pyplot as plt 
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import io
from dataset.CubiCasaDatset import CubiCasaDatset, cubicasa_openings_dict

# Load dataset to memory
ds = CubiCasaDatset(image_channels=3)

# List for openings labels
cubicasa_openings_dict_labels = [str.encode(x) for x in cubicasa_openings_dict]

def create_tf_example(item):
    
    """
    Generate tf.Example for item
    
    Parameters
    ----------
    item : object
        CubiCasa item
    
    Output
    ------
    tf.data.Example
    """

    height, width = item['image_original_shape'][0], item['image_original_shape'][1]
    filename = item['image_path']
    
    encoded_image_data = io.BytesIO()
    image_data = Image.fromarray(np.uint8(item['image'].numpy()))
    image_data.save(encoded_image_data, format='PNG')
    encoded_image_data = encoded_image_data.getvalue()

    filename = str.encode(filename) # to bytes
    image_format = b'png'
    
    # Openings_bbox = (x_min,y_min,height,width)
    xmins = item['openings_bbox'].numpy()[:,0] / width
    xmaxs = (item['openings_bbox'].numpy()[:,0] + item['openings_bbox'].numpy()[:,2])  / width 
    ymins = item['openings_bbox'].numpy()[:,1] / height
    ymaxs = (item['openings_bbox'].numpy()[:,1] + item['openings_bbox'].numpy()[:,3]) / height

    # Make sure boxes are in picture
    xmins[xmins < 0 ] = 0
    xmins[xmins > 1 ] = 1
    xmaxs[xmaxs < 0 ] = 0
    xmaxs[xmaxs > 1 ] = 1

    ymins[ymins < 0 ] = 0
    ymins[ymins > 1 ] = 1
    ymaxs[ymaxs < 0 ] = 0
    ymaxs[ymaxs > 1 ] = 1

    # classes 
    classes_text = list(map(lambda x: cubicasa_openings_dict_labels[x-1], item['openings_class'].numpy()))
    classes = list(item['openings_class'].numpy()) # List of integer class id of bounding box (1 per box)
    
    # create tf.data.Example 
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

# Define train/val indices    
num_samples = len(ds.data)
indices = np.arange(num_samples)
train_idx, val_idx = train_test_split(indices, test_size=0.2, train_size=None)

# Write training data
writer = tf.io.TFRecordWriter('train.tfrecord')
for i in train_idx:
    tf_example = create_tf_example(ds.get_sample(i,include_openings=True))
    writer.write(tf_example.SerializeToString())
writer.close()

# Write validation data
writer = tf.io.TFRecordWriter('val.tfrecord')
for i in val_idx:
    tf_example = create_tf_example(ds.get_sample(i,include_openings=True))
    writer.write(tf_example.SerializeToString())
writer.close()
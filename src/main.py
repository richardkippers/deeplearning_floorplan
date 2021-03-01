"""
Test Train 
@author r.kippers, 2019 
""" 

from dataset.CubiCasaDatset import CubiCasaDatset
from deeplearning.semantic_segmentation.UNet import UNetModelBuilder
from deeplearning.evaluation.iou import iou

import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import matplotlib.pyplot as plt 
import numpy as np

# Globals
LOW_GPU_MEMORY = True

# Load data 

ds = CubiCasaDatset()
print(len(ds.data), "data samples")

# Display first floor plan + mask
sample_0 = ds.get_sample(0)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(sample_0["folder_path"])
ax1.imshow(sample_0["image"].numpy()[:,:,0],cmap='gray')
ax2.imshow(sample_0["mask"].numpy()[:,:,0],cmap='gray')
plt.show()

## Hyperparameters

# batch = 8
# lr = 1e-4

# train_dataset = ds.get_tf_dataset("training")
# valid_dataset = ds.get_tf_dataset("validation")

# mb = UNetModelBuilder(LOW_GPU_MEMORY)
# model = mb.build_model()

# opt = tf.keras.optimizers.Adam(lr)
# metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
# model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

# callbacks = [
#     #ModelCheckpoint("files/model.h5"),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
#     #CSVLogger("files/data.csv"),
#     #TensorBoard(),
#     EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
# ]


# train_steps = 100
# valid_steps = 100
# epochs = 20

# print(model.summary())

# model.fit(train_dataset,
#     validation_data=valid_dataset,
#     epochs=epochs,
#     steps_per_epoch=train_steps,
#     validation_steps=valid_steps,
#     callbacks=callbacks)
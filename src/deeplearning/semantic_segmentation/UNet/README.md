# UNet 

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.


### Training
```
from dataset.CubiCasaDatset import CubiCasaDatset

from deeplearning.semantic_segmentation.UNet.UNet import UNetModelBuilder
from deeplearning.evaluation.iou import iou

import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import matplotlib.pyplot as plt 
import os, datetime

ds = CubiCasaDatset(image_channels=3)


# Init model for training
mb = UNetModelBuilder(False)
model = mb.build_model()

lr = 1e-4
batch = 8

train_dataset = ds.get_tf_dataset_wall_mask("training", batch=batch)
valid_dataset = ds.get_tf_dataset_wall_mask("validation", batch=batch )


opt = tf.keras.optimizers.Adam(lr)
metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [
    ModelCheckpoint("unet/model.h5"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    #CSVLogger("files/data.csv"),
    tensorboard_callback,
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]

train_steps = 100
valid_steps = 20
epochs = 80

# Do train 

model.fit(train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=callbacks)

# View result

predict_image = ds.get_sample(1240)['image'].numpy()

y_pred = model.predict(np.expand_dims(predict_image, axis=0))[0] > 0.5

sample_0 = ds.get_sample(0)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(sample_0["folder_path"])
ax1.imshow(predict_image[:,:,0],cmap='gray')
ax2.imshow(y_pred[:,:,0],cmap='gray')
plt.show()


```

# RetinaNet 

Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).

I did use a lot of code from https://keras.io/examples/vision/retinanet to implement RetinaNet

## Training


```
from dataset.CubiCasaDatset import CubiCasaDatset
import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import matplotlib.pyplot as plt 
import numpy as np
import socket 
import os
from deeplearning.object_detection.RetinaNet.LabelEncoder import LabelEncoder
from deeplearning.object_detection.RetinaNet.ResNet50_backbone import get_backbone
from deeplearning.object_detection.RetinaNet.loss import RetinaNetLoss
from deeplearning.object_detection.RetinaNet.RetinaNet import RetinaNet
from deeplearning.object_detection.RetinaNet.DecodePredictions import DecodePredictions

ds = CubiCasaDatset(image_channels=3)
print(len(ds.data), "data samples")

# Display first floor plan + mask
sample_0 = ds.get_sample(2,include_wall_mask=True, include_openings=True)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(sample_0["folder_path"])
ax1.imshow(sample_0["image"].numpy()[:,:,0],cmap='gray')
ax2.imshow(sample_0["wall_mask"].numpy()[:,:,0],cmap='gray')

for i in sample_0["openings_bbox"]: 
    x1, y1, w, h = i[0], i[1], i[2], i[3]
    patch = plt.Rectangle(
        [x1, y1], w, h, fill=False, edgecolor="blue", linewidth=1
    )
    ax2.add_patch(patch)

plt.show()



model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 2
batch_size = 10

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]

learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]


train_dataset = ds.get_tf_dataset_openings("training") 
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch #, num_parallel_calls=autotune   
)

val_dataset = ds.get_tf_dataset_openings("validation") 
val_dataset = val_dataset.shuffle(8 * batch_size)
val_dataset = val_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(
    label_encoder.encode_batch #, num_parallel_calls=autotune
)

epochs = 30

model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)
# Inference model
image = tf.keras.Input(shape=[None, None, 3], name="image")

latest_checkpoint = tf.train.latest_checkpoint(model_dir)
model.load_weights(latest_checkpoint)

predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

def display_prediction(val_sample, detections):
  # Display prediction
  fig, (ax1, ax2) = plt.subplots(1, 2)
  num_detections = detections.valid_detections[0]
  fig.suptitle(val_sample["folder_path"] + ", " + str(num_detections) +  " detections")
  ax1.imshow(val_sample["image"].numpy()[:,:,0],cmap='gray')
  ax2.imshow(val_sample["image"].numpy()[:,:,0],cmap='gray')

  # ground truth
  for i in val_sample["openings_bbox"]: 
      x1, y1, w, h = i[0], i[1], i[2], i[3]
      patch = plt.Rectangle(
          [x1, y1], w, h, fill=False, edgecolor="blue", linewidth=1
      )
      ax1.add_patch(patch)

  # prediction
  for bbox in detections.nmsed_boxes[0][:num_detections]:
      x1, y1, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
      patch = plt.Rectangle(
          [x1, y1], w, h, fill=False, edgecolor="blue", linewidth=1
      )
      ax2.add_patch(patch)

  plt.show()


# Display result
for i in range(300):
  val_sample = ds.get_sample(i, include_wall_mask=False, include_openings=True)
  val_image = tf.cast(val_sample["image"], dtype=tf.float32)
  val_image = tf.keras.applications.resnet.preprocess_input(val_image)
  val_image = tf.expand_dims(val_image, axis=0)
  detections = inference_model.predict(val_image)
  num_detections = detections.valid_detections[0]

  if num_detections > 0: 
    print("i, ", i)
    print("Num detections:", num_detections)
    display_prediction(val_sample, detections)

```
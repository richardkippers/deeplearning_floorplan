# Deep Floor Plan Reconstruction using the CubiCasa5K Data Set

Using this for my masters thesis at the University of Twente. The goal is to obtain the following information: wall, opening (window or door), room (including room type), and the front door.

![General Idea](general_floor_plan_dl.png)


### Contents 

* src / 
    * dataset (input files) 
    * models (TF/Keras Models)
    * readers (data readers)
    * main.py

### Requirements and conda environment
```
conda env create -f environment.yml
source activate deeplearning_floorplan
```

### Load data 

The dataset [Kalervo et al., 2019](https://arxiv.org/abs/1904.01920) is used as training data. The original [GitHub repo](https://github.com/CubiCasa/CubiCasa5k) is a PyTorch implementation.

```
cd input
wget -O input/cubicasa5k.zip https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1
! unzip -qq cubicasa5k.zip #-qq for quiet
```


### GPU installation 

[Tensorflow GPU Guide](https://www.tensorflow.org/install/gpu), or just run on [Google Colab](https://colab.research.google.com)

### View data 
To view a data sample, run the following Python script: 

```
from dataset.CubiCasaDatset import CubiCasaDatset
import matplotlib.pyplot as plt 

ds = CubiCasaDatset()
print(len(ds.data), "data samples")

sample_0 = ds.get_sample(0,include_wall_mask=True, include_openings=True)

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
```

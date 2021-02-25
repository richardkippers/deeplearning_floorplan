# Deep Floor Plan Reconstruction using the CubiCasa5K Data Set

Using this a part for my masters thesis at the University of Twente


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

```
wget -O input/cubicasa5k.zip https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1
! unzip -qq cubicasa5k.zip #-qq for quiet
```

### GPU installation 

https://www.tensorflow.org/install/gpu
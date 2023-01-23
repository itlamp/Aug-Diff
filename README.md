# Aug-Diff
By Itay Lamprecht and Eran Avneri

This repository is part of a final project in the Technion's course 046211 - Deep Learning

![alt text](https://github.com/itlamp/Aug-Diff/blob/main/assets/intro_pic.JPG?raw=True)

# Generative augmentations using diffusion to improve performance of NN's

In our project, we aim to demonstrate the effectiveness of generative augmentation using diffusion models on small datasets.

We train a classification CNN over the STL dataset in multiple settings:

 - Original data: STL dataset
 - Traditional Augmentations: STL dataset using standard augmentations
 - Generative Augmentations: STL dataset and a varying percentage of the generated data - created using stable diffusion
 - Generative Augmentations and traditional augmentations: STL dataset, the generated data, with traditional data augmentations.
 - Only generated:  training solely on the generated data.

## Prerequisites
 - After cloning into the repository, please run:
    
    ```
    conda env create -f environment.yml
    conda activate augdiff
    ```

## Datasets
STL dataset:
 - The stl dataset was downloaded using the help of: https://github.com/mttk/STL10/. 
 
 Generated images:
 -  Images generated using code in the repository https://github.com/CompVis/stable-diffusion, using plms sampler and v1.4 checkpoint.

To use our code, please make sure that all of the images are in a file with the corresponding name. For example, if the stl images in the train set that are saved in <stl_train_path>, images of dogs should be at '<stl_train_path>/dog. Same goes for generated images, images of airpanes should be at '<gen_path>/airplane'. The labels are:

```
'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'
```

## Training:

With conda env installed and datasets created, run:
    python Main.py
With the following arguments:
### Required arguments:

--gen_path - path to generated images

--stl_train_path - path to train set

--stl_test_path - path to test set

### Optional arguments:
--only_gen - If chosen, then uses only generated images in training the network. If False, uses both generated and original images. Default is False.

--mix_pct - How many generated images to use. 0 corresponds to using no generated images, while 1 corresponds to doubling the size of the original dataset with generated images. Must be a number between 0 and 1. Default is 1.

--augment - If chosen, Classical augmentations are used as defined int the file get_transforms.py. Otherwise, no augmentations are used. Default is true.

### Example
To run with 250 additional generated images, without classical augmentations, run:

```
python Main.py --gen_path <path> --stl_train_path <path> --stl_test_path <path> --mix_pct 0.5 --augment 
```

## Results
Our main results are illustrated in the following graph. We show that adding more augmentations to the original dataset enhances performence on the test set.

![alt text](https://github.com/itlamp/Aug-Diff/blob/main/assets/Results.png?raw=True)

## Credits:
Other than repositories previously acknowledged:
 - The model being trained is based on a model presented in our TA's github for the Technions 046211 course - https://github.com/taldatech/ee046211-deep-learning. Specifically, from tutorial number 6.

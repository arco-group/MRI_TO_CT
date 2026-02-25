# MRI_TO_CT
Collection of 10 Generative Adversarial Network varieties 

# From MRI to Virtual CT: Development of an MRI-only Workflow for Synthetic Dose Planning
This repository contains the code used to run experiments related to MRI-to-synthetic CT translation.

# Organization of the code 
The code is organized into three folders:
1. "configs": contains the YAML files used to set the experiment parameters. For each model, the training script     "train_kfold.py" loads the "{modelname}_train.yaml" configuration file, while the testing script "test.py" loads the            "{modelname}_test.yaml" configuration file.
2. "data": Example folder that specifies the path referenced in the "configs" files to access the dataset used for training.
In this case, the path is data/k_fold_cross_validation/folds_2d/folds_2d_AB.
Inside the "folds_2d_AB" directory, there are five folders corresponding to the five-fold cross-validation for the abdominal anatomical district (AB). Within each fold directory, .CSV files are provided containing the relative paths to the dataset images.
3. "src": contains "model" and "utils" folders. The "model" folder includes subfolders for Bicyclegan, Cogan, Cyclegan, Discogan, Dualgan, Munit, PixPix, PixelDA, Stargan and Unit, each containing specific model code. The "utils" folder includes the "util_general.py", it allows saving and loading model checkpoints (including model weights and optimizer state), both during training and testing, with the option to remap the device and update the learning rate, and the "util_data.py" code for data loading and preprocessing utilities for an MRI-to-CT deep learning workflow.

For each model (Bicyclegan, Cogan, Cyclegan, Discogan, Dualgan, Munit, PixPix, PixelDA, Stargan and Unit), you can run training or testing code. These codes are located in the "src -> model -> Bicyclegan/Cogan/Cyclegan/Discogan/Dua/Munit/ PixPix/PixelDA/Stargan/Unit" folders.

1. Running the training code trains the model on the public dataset(SynthRAD2025). Experiment parameters can be set using the configuration files "bicyclegan_train.yaml", "cogan_train.yaml","cyclegan_train.yaml", "discogan_train.yaml", "dualgan_train.yaml", "munit_train.yaml", "pix2pix_train.yaml", "pixelda_train.yaml", "stargan_train.yaml" or "unit_train.yaml".
2. Running the test code tests the trained model on the desired dataset. Experiment parameters, including the test dataset, can be set using the configuration files "bicyclegan_test.yaml", "cogan_test.yaml","cyclegan_test.yaml", "discogan_test.yaml", "dualgan_test.yaml", "munit_test.yaml", "pix2pix_test.yaml", "pixelda_test.yaml", "stargan_test.yaml" or "unit_test.yaml".

# Contact
For questions and comments, feel free to contact: alessandro.pesci@unicampus.it, valerio.guarrasi@unicampus.it

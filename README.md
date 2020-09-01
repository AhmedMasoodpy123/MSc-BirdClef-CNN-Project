# MSc-BirdClef-CNN-Project

## Introduction:
The accurate classification of the identity  of bird species is essential for most biodiversity projects.
In particular, groups such as ornithologists and ecological consultants are potential users of an automated bird sound identifying system, specifically for conservation or surveillance initiatives. The LifeCLEF foundation have setup the BirdCLEF initiative to evaluate various audio based bird identification systems.

In order to run the nets, we first need to install different modules. For this run the following command on your terminal:

pip3 install -r requirements.txt

This will install all the required libraries on your system. 

The project consists of 3 tasks:

1- Preprocessing

2- Building the Dataset

3- Training and Evaluation

# Preprocessing
For this, we are using the Data_Preprocessing.ipynb file. As a prerequisite to the file, please create a folder with all the bird files.
Bird audio files are available from the following sources:

1- https://www.xeno-canto.org

2-Specificially for this project: https://www.imageclef.org/BirdCLEF2020

If this first data source is used, bird sounds to be placed in folders correlating to the species.
Once this is done,  the jupyter notebook Data_Preprocessing.ipynb can be run to generate results and see graphical comparison.
This will run the file and produce the preprocessed files in an output folder.

# Building the Dataset:
Once we have the preprocessed files, we need to divide the data in train:validation:test data group but before this we need to get the spectrograms of each mp3 file. For this we run the build_dataset.ipynb and it will save all the spectograms on a drive

# Train and Evaluation:
In this phase, we utilise params.json files found in the hyperparameters folder.
A valid json_format is as follows:
        
        {
            "model": 1,
            "width": 128,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "num_epochs": 120,
            "save_summary_steps": 100,
            "num_workers": 4,
            "growthRate": 12, 
            "depth": 15, 
            "reduction": 0.5,
            "optimizer": 2,
            "threshold": 0.5,
            "dropout": 0.5
        }

Once we have the params and we run the ipynb file, it will start training for that particular model and results will be plotted.
BirdClefs metric is mAP which is defined in the net.ipynb file and can be called during training and evaluation.

# Altering conv layers to FST type
To modify convolutional layers to fit FST requirements, Alter kernel size in the following way:

        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=(1,1),
                               bias=False)

should change to:

        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size,
                               bias=False)

Rerun training steps for each model if you would like to see comparison.




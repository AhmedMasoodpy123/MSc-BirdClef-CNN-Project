# MSc-BirdClef-CNN-Project

## Data Preprocessing

We are tackling the problem of recognizing individual bird species in mixtures of sounds. We can synthesize datasets by superimposing various sound clips. Therefore, we will pre-process our raw data--audios, by transforming them into spectrograms. The following command is run.

    python3 aud_to_spec.ipynb --src_dir <path_to_raw_audios> --spec_dir <path_to_spec_destination>

## Building the dataset

Using the folliwng code, one is able to split the data between the training, validation and test datasets through implementation of the build_dataset notebook.

    python3 build_dataset.ipynb --data_dir <path_to_preprocessed_data> --output_dir <path_to_desired_splitted_datasets>
    
## Train the Model

First, create a .json file that sets teh parameters for yoru neural net. 
Then Run the follwing line of code:

    python3 train.ipynb --data_dir <path_to_splitted_datasets> --model_dir <path_to_the_folder_of_json_file>
    
## Evaluation

To compare all neural nets, one uses the following code.

    python3 synthesize_results.ipynb --parent dir <path-to-parent-folder-with-various-exps>

To evaluate the tests set, one uses the following code:

    python evaluate.py --data_dir <path-to-test-data> --model_dir <path-to-folder-of-the-selected-model>
    


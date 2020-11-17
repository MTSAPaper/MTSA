# MASA: Multimodal Alignment for Sentiment Analysis
This repository contains the code for MASA.

### All of the dependencies are in the requirements.txt. You can build it with the following command:

`conda create --name <env> --file requirements.txt`

### Instructions
We provide the default hyperparameters in params.py, you can modify it to try different hyperparamters. To run the code in this folder, please follow the instructions below.

1. Set the data_path in param.py to the path of MOSI or MOSEI dataset.
2. Set the save_path in param.py to to any path you perfer to save the ckeckpoint and results.
3. Run `python run.py`

To load the checkpoint or results, please follow the instructions below.
1. Set the ckpt_path to the path of checkpoint you want to load.
2. Set the res_path to the path of results you want to load or write.
3. Run `python stats.py`

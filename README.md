# ArduProg
A Replication Package for ArduProg

# Dataset and Labelling Results
We release the dataset used for the experiments in our paper [here](https://zenodo.org/record/7256145#.Y1oiJXZByUk). Moreover, the labelling results can be found inside the `labelling_results` folder.

# Environment Setting
Run the following commands:
1. pip install -r requirements.txt
2. docker build dockerfile --tag arduprog:1.0
3. docker run -it arduprog:1.0 /bin/bash
4. download this repository, then enter unzip the repository and enter the directory
5. download the dataset [here](https://zenodo.org/record/7256145#.Y1oiJXZByUk), then put it in this directory

# Running Training
1. cd src/insert-component-name-you-want-to-train
2. python train.py

Optionally, you can change the model and training setting by modifying the dictionary values inside `config.py`


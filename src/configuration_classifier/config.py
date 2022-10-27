import os
from datetime import datetime

base_config = {
    # model
    'model_path': 'microsoft/codebert-base',
    
    # dataset_path
    'dataset_path': '../../data_training/config_classifier/train.csv',

    # training config
    'num_epochs': 30,
    'learning_rate': 5e-5,
    'batch_size': 32,
    'gradient_accumulation_steps': 1,
    'learning_rate': 2e-5,
}

def initialize_config(config):
    config_cp = config.copy()
    config_cp['output_dir'] = f'output/{config_cp["model_path"].split("/")[-1]}-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(config_cp['output_dir'], exist_ok=True)

    return config_cp
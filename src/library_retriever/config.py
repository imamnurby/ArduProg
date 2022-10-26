from datetime import datetime
import os

base_config = {
    'model_path': 'codebert',    # bert, codebert, distillbert, roberta
    
    # training configuration
    'num_epochs': 50,
    'batch_size': 256,
    'learning_rate': 2e-5,
    'warmup_steps': 1000,
    'use_amp': True,
    'checkpoint_save_total_limit': 1,
    
    # dataset config
    'training_data_path': '../../data_training/library_retriever/train.csv'

}

def initialize_config(config):
    config_cp = config.copy()
    config_cp['model_save_path'] = f'output/{config_cp["model_path"]}-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(config_cp['model_save_path'], exist_ok=True)

    config_cp['checkpoint'] = config_cp['model_save_path'] + '/checkpoint'

    if config['model_path'] == 'codebert':
        config_cp['model_path'] = 'microsoft/codebert-base'
    
    elif config['model_path'] == 'bert':
        config_cp['model_path'] = 'bert-base-uncased'
    
    elif config['model_path'] == 'roberta':
        config_cp['model_path'] = 'roberta-base'

    elif config['model_path'] == 'distillbert':
        config_cp['model_path'] = 'distilbert-base-uncased'

    else:
        raise ValueError("Undefined model path")

    return config_cp
from datetime import datetime
import os

base_config = {
    # model choice
    'model': 'plbart', # roberta, codet5, codebert
    'checkpoint_path': None,
    
    # model config
    'num_beams': 1,
    'return_top_k': 1,
    'max_input_len': 150,
    'max_target_len': 150,

    # dataset
    'dataset_path': '../../data_training/pattern_generator/train.csv',
    
    # training config
    'epochs': 100,
    'batch_size': 64,
    'gradient_accumulation_step': 4,
    'warmup_steps': 1000,
    'learning_rate': 5e-5,
}

def initialize_config(config):
    config_cp = config.copy()
    config_cp['output_dir'] = f'output/{config_cp["model"]}-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(config_cp['output_dir'], exist_ok=True)

    return config_cp
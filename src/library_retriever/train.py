from torch.utils.data import DataLoader
from sentence_transformers import losses, LoggingHandler
from config import base_config, initialize_config
import logging
import pandas as pd
from datasets import Dataset

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

# load config
config = initialize_config(base_config)

#Loading model
model = SentenceTransformerCustom(config['model_path'])

# load dataset
logger.info(f"dataset loaded from: {config['training_data_path']}")
dataset = Dataset.from_pandas(
    pd.read_csv(config['training_data_path'])
)

# create data loader and loss for OnlineContrastiveLoss
train_dataloader_ConstrativeLoss = DataLoader(dataset, 
                                            shuffle=True, 
                                            batch_size=config['batch_size'], 
                                            drop_last=True, 
                                            pin_memory=True, 
                                            num_workers=3)

train_loss_ConstrativeLoss = losses.TripletLoss(model=model)

# train the model
model.fit(train_objectives=[(train_dataloader_ConstrativeLoss, train_loss_ConstrativeLoss)],
          epochs=config['num_epochs'],
          warmup_steps=config['warmup_steps'],
          output_path=config['model_save_path'],
          use_amp=config['use_amp'],
          checkpoint_save_total_limit=config['checkpoint_save_total_limit'],
          optimizer_params = {'lr': config['learning_rate']},
          checkpoint_path=config['checkpoint']
          checkpoint_save_steps = len(train_dataloader_ConstrativeLoss),
          )
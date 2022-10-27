from transformers import RobertaForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import datasets
import pandas as pd
import torch
from config import base_config, initialize_config

# load config
config = initialize_config(base_config)

# load model
tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
model = RobertaForSequenceClassification.from_pretrained(
    config['model_path'], 
    num_labels=4,
    problem_type="multi_label_classification"
)

# load dataset
dataset = pd.read_csv(config['dataset_path'])
train = dataset.sample(n=255, random_state=1234)
valid = dataset[~dataset.index.isin(train.index)]
assert(len(train)+len(valid)==len(dataset))

dataset = datasets.DatasetDict(
    {
        'train': datasets.Dataset.from_pandas(train),
        'valid': datasets.Dataset.from_pandas(valid)
    }
)

def preprocess_function(row):
    model_inputs = tokenizer(row['features'], truncation=True, padding=False)
    temp_list = [row['is_uart'], row['is_spi'], row['is_i2c'], row['is_none']] # create a temp list containing the label
    labels =  torch.FloatTensor(temp_list) # convert the list to torch.Foat datatype as the model only recognize this type
    model_inputs["labels"] = labels
    return model_inputs

dataset = dataset.map(
    preprocess_function, 
    remove_columns=dataset['train'].column_names
)

# training
args = TrainingArguments(
    f"{config['output_dir']}",
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    do_train=True,
    do_eval=True,
    learning_rate=config['learning_rate'],
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    weight_decay=0.0,
    warmup_steps=1000,
    save_total_limit=1,
    num_train_epochs=config['num_epochs'],
    fp16=True,
    optim='adamw_torch',
    group_by_length=True)


trainer = Trainer(
    model,
    args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer
)

trainer.train()





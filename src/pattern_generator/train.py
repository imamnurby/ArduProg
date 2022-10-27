from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import (AutoTokenizer, EncoderDecoderModel, T5ForConditionalGeneration, PLBartForConditionalGeneration)
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from typing import Optional, Any, Union
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd
import os
import copy
from config import base_config, initialize_config

# helper
def preprocess_function(examples):
    inputs = [ex for ex in examples["source"]]
    targets = [ex for ex in examples["target"]]

    model_inputs = tokenizer(inputs, max_length=config['max_input_len'], truncation=True, padding=False)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=config['max_target_len'], truncation=True, padding=False)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

@dataclass
class DataCollatorForSeq2Seq:
    """
    refer to the original documentation from HuggingFace
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        return features

def instantiate_model(config):
    if config['model'] == 'codebert':
        tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        model = EncoderDecoderModel.from_encoder_decoder_pretrained('microsoft/codebert-base', 'microsoft/codebert-base')
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.architectures = 'EncoderDecoderModel'
        model.config.max_length = config['max_input_len']

    elif config['model'] == 'codet5':
        tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

    elif config['model'] == 'plbart':
        tokenizer = AutoTokenizer.from_pretrained('uclanlp/plbart-csnet')
        model = PLBartForConditionalGeneration.from_pretrained('uclanlp/plbart-csnet')
    
    else:
        raise ValueError("Undefined model path")

    return tokenizer, model

# load config
config = initialize_config(base_config)

# load tokenizer and model
tokenizer, model = instantiate_model(config)

# preprocess dataset
df = pd.read_csv(config['dataset_path'])
val = df.sample(n=500, random_state=1234).copy()
train = df[~df.index.isin(val.index)]
dataset = DatasetDict(
        {
            'train': Dataset.from_pandas(train),
            'val': Dataset.from_pandas(val)
        }
    )

dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

# training
args = Seq2SeqTrainingArguments(
    f'{config["output_dir"]}',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    do_train=True,
    do_eval=False,
    learning_rate=config['learning_rate'],
    per_device_train_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_step'],
    weight_decay=0.0,
    warmup_steps=config['warmup_steps'],
    save_total_limit=1,
    num_train_epochs=config['epochs'],
    fp16=True,
    optim='adamw_torch')

# train model
trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer
    )

trainer.train()





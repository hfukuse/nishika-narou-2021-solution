import warnings

import pandas as pd
import re
import os
import argparse
from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
import json

warnings.filterwarnings('ignore')
os.system('pip install transformers fugashi ipadic unidic_lite --quiet')


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", default=False, action="store_true", help="debug")
    arg("--settings", default="./nishika-narou-2021-solution/settings.json", type=str, help="settings path")
    arg("--is_test", action="store_true", help="test")
    return parser

args = make_parse().parse_args()

with open(args.settings) as f:
    js = json.load(f)


class Config:
    model_name = js["model_name"]
    pretrained_model_path = js["pretrained_model_path"]
    batch_size = js["batch_size"]
    device = js["device"]
    seed = js["seed"]
    train_dir = js["train_dir"]
    dataset_dir = js["dataset_dir"]
    narou_dir = js["narou_dir"]
    models_dir = js["models_dir"]


import sys

sys.path.append(Config.narou_dir + "/utils")

from preprocess import remove_url


def main():
    train_data = pd.read_csv(Config.train_dir + '/kfold_2021_07.csv')
    test_data = pd.read_csv(Config.dataset_dir + '/test.csv')

    train_data = train_data.drop(columns=['fold'])

    data = pd.concat([train_data, test_data])
    data["excerpt"] = data["story"]
    data['excerpt'] = data['excerpt'].apply(lambda x: x.replace('\n', ''))

    data.excerpt = data.excerpt.map(remove_url)

    text = '\n'.join(data.excerpt.tolist())

    model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    tokenizer.save_pretrained(Config.pretrained_model_path)

    with open(Config.pretrained_model_path + '/text.txt', 'w') as f:
        f.write(text)

    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=Config.pretrained_model_path + "/text.txt",
        block_size=256)

    valid_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=Config.pretrained_model_path + "/text.txt",
        block_size=256)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    num_train_epochs = 3
    eval_steps = 400
    if args.debug:
        num_train_epochs = 1
        eval_steps = 2169

    training_args = TrainingArguments(
        output_dir=Config.pretrained_model_path + "/bert_base_chk",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy='steps',
        save_total_limit=2,
        eval_steps=eval_steps,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        prediction_loss_only=True,
        report_to="none")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset)

    trainer.train()
    trainer.save_model(Config.pretrained_model_path)


if __name__ == "__main__":
    main()

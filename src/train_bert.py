import os
import random
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoConfig
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

plt.style.use("seaborn-talk")
from time import time
from colorama import Fore, Back, Style

r_ = Fore.RED
b_ = Fore.BLUE
g_ = Fore.GREEN
y_ = Fore.YELLOW
w_ = Fore.WHITE
bb_ = Back.BLACK
sr_ = Style.RESET_ALL


class Config:
    model_name = "cl-tohoku/bert-base-japanese-v2"
    pretrained_model_path = "./bert-base-japanese"
    output_hidden_states = True
    epochs = 2
    batch_size = 8
    device = "cuda"
    seed = 42
    max_len = 256
    lr = 1e-5
    wd = 0.01
    eval_schedule = [(float("inf"), 40), (0.85, 30), (0.80, 20), (0.70, 10), (0, 0)]
    gradient_accumulation = 2
    train_dir = "./train_val_split"
    item = ["story", "title", "keyword"]
    item_num = 0  # 0ならstory(あらすじ),1ならtitle(題名),2ならkeyword(タグ)
    dataset_dir = "../input/nishika-narou"
    models_dir = "models/n8_model"


os.system("pip install transformers fugashi ipadic unidic_lite --quiet")
os.system("mkdir -p " + Config.models_dir)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed=Config.seed)

kfold_df = pd.read_csv(Config.train_dir + "/kfold_2021_06.csv")

kfold_df["excerpt"] = kfold_df[Config.item[Config.item_num]]
kfold_df.excerpt = kfold_df.excerpt.replace("\n", "", regex=True)
import re


def remove_url(sentence):
    ret = re.sub(r"(http?|ftp)(:\/\/[-_\.!~*\"()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", str(sentence))
    return ret


kfold_df.excerpt = kfold_df.excerpt.map(remove_url)

kfold_df["target"] = kfold_df["fav_novel_cnt_bin"]

from torch.utils.data import Dataset


def convert_examples_to_features(text, tokenizer, max_len):
    tok = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    return tok


class CLRPDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.excerpts = self.data.excerpt.tolist()
        if not is_test:
            self.targets = self.data.target.tolist()

        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if not self.is_test:
            excerpt = self.excerpts[item]
            label = self.targets[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer, self.max_len
            )
            return {
                "input_ids": torch.tensor(features["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(features["attention_mask"], dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.float),
            }
        else:
            excerpt = self.excerpts[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer, self.max_len
            )
            return {
                "input_ids": torch.tensor(features["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(features["attention_mask"], dtype=torch.long),
            }


import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, h_size, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(h_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class CLRPModel(nn.Module):
    def __init__(self, transformer, config):
        super(CLRPModel, self).__init__()
        self.h_size = config.hidden_size
        self.transformer = transformer
        self.head = AttentionHead(self.h_size * 4)
        self.linear = nn.Linear(self.h_size * 2, 5)
        self.linear_out = nn.Linear(self.h_size * 8, 5)

    def forward(self, input_ids, attention_mask):
        transformer_out = self.transformer(input_ids, attention_mask)

        all_hidden_states = torch.stack(transformer_out.hidden_states)
        cat_over_last_layers = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
        )

        cls_pooling = cat_over_last_layers[:, 0]
        head_logits = self.head(cat_over_last_layers)
        y_hat = self.linear_out(torch.cat([head_logits, cls_pooling], -1))
        return y_hat


def create_optimizer(model):
    named_parameters = list(model.named_parameters())

    roberta_parameters = named_parameters[:197]  # 389
    attention_parameters = named_parameters[199:203]  # 391:395
    regressor_parameters = named_parameters[203:]  # 395

    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})

    increase_lr_every_k_layer = 1
    lrs = np.linspace(1, 5, 24 // increase_lr_every_k_layer)
    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        splitted_name = name.split(".")
        lr = Config.lr
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[3]):
            layer_num = int(splitted_name[3])
            lr = lrs[layer_num // increase_lr_every_k_layer] * Config.lr

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return optim.AdamW(parameters)


class DynamicPadCollate:
    def __call__(self, batch):

        out = {"input_ids": [],
               "attention_mask": [],
               "label": []
               }

        for i in batch:
            for k, v in i.items():
                out[k].append(v)

        max_pad = 0

        for p in out["input_ids"]:
            if max_pad < len(p):
                max_pad = len(p)

        for i in range(len(batch)):
            input_id = out["input_ids"][i]
            att_mask = out["attention_mask"][i]
            text_len = len(input_id)

            out["input_ids"][i] = (out["input_ids"][i].tolist() + [1] * (max_pad - text_len))[:max_pad]
            out["attention_mask"][i] = (out["attention_mask"][i].tolist() + [0] * (max_pad - text_len))[:max_pad]

        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long)
        out["label"] = torch.tensor(out["label"], dtype=torch.float)

        return out


class AvgCounter:
    def __init__(self):
        self.reset()

    def update(self, loss, n_samples):
        self.loss += loss * n_samples
        self.n_samples += n_samples

    def avg(self):
        return self.loss / self.n_samples

    def reset(self):
        self.loss = 0
        self.n_samples = 0


class EvaluationScheduler:
    def __init__(self, evaluation_schedule, penalize_factor=1, max_penalty=8):
        self.evaluation_schedule = evaluation_schedule
        self.evaluation_interval = self.evaluation_schedule[0][1]
        self.last_evaluation_step = 0
        self.prev_loss = float("inf")
        self.penalize_factor = penalize_factor
        self.penalty = 0
        self.prev_interval = -1
        self.max_penalty = max_penalty

    def step(self, step):
        if step >= self.last_evaluation_step + self.evaluation_interval:
            self.last_evaluation_step = step
            return True
        else:
            return False

    def update_evaluation_interval(self, last_loss):
        cur_interval = -1
        for i, (loss, interval) in enumerate(self.evaluation_schedule[:-1]):
            if self.evaluation_schedule[i + 1][0] < last_loss < loss:
                self.evaluation_interval = interval
                cur_interval = i
                break

        self.prev_loss = last_loss
        self.prev_interval = cur_interval


def make_dataloader(data, tokenizer, is_train=True):
    dataset = CLRPDataset(data, tokenizer=tokenizer, max_len=Config.max_len)
    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    batch_dataloader = DataLoader(dataset, sampler=sampler, batch_size=Config.batch_size, pin_memory=True,
                                  collate_fn=DynamicPadCollate())
    return batch_dataloader


class Trainer:
    def __init__(self, train_dl, val_dl, model, optimizer, scheduler, scaler, criterion, model_num):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = Config.device
        self.batches_per_epoch = len(self.train_dl)
        self.total_batch_steps = self.batches_per_epoch * Config.epochs
        self.criterion = criterion
        self.model_num = model_num
        self.scaler = scaler

    def run(self):
        record_info = {
            "train_loss": [],
            "val_loss": [],
        }

        best_val_loss = float("inf")
        evaluation_scheduler = EvaluationScheduler(Config.eval_schedule)
        train_loss_counter = AvgCounter()
        step = 0

        for epoch in range(Config.epochs):
            print(f"{r_}Epoch: {epoch + 1}/{Config.epochs}{sr_}")
            start_epoch_time = time()

            for batch_num, batch in enumerate(self.train_dl):
                train_loss = self.train(batch, step)
                train_loss_counter.update(train_loss, len(batch))
                record_info["train_loss"].append((step, train_loss.item()))

                if evaluation_scheduler.step(step):
                    val_loss = self.evaluate()

                    record_info["val_loss"].append((step, val_loss.item()))
                    print(
                        f"\t\t{epoch + 1}#[{batch_num + 1}/{self.batches_per_epoch}]: train loss - {train_loss_counter.avg()} | val loss - {val_loss}", )
                    train_loss_counter.reset()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss.item()
                        print(f"\t\t{g_}Val loss decreased from {best_val_loss} to {val_loss}{sr_}")
                        torch.save(self.model, Config.models_dir / f"best_model_{self.model_num}.pt")

                    evaluation_scheduler.update_evaluation_interval(val_loss.item())

                step += 1
            end_epoch_time = time()
            print(f"{bb_}{y_}The epoch took {end_epoch_time - start_epoch_time} sec..{sr_}")

        return record_info, best_val_loss

    def train(self, batch, batch_step):
        self.model.train()
        sent_id, mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch[
            "label"].to(self.device)
        with autocast():
            preds = self.model(sent_id, mask)
            train_loss = self.criterion(preds, labels)

        self.scaler.scale(train_loss).backward()

        if (batch_step + 1) % Config.gradient_accumulation or batch_step + 1 == self.total_batch_steps:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.zero_grad()
        self.scheduler.step()
        return train_loss

    def evaluate(self):
        self.model.eval()
        val_loss_counter = AvgCounter()

        for step, batch in enumerate(self.val_dl):
            sent_id, mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch[
                "label"].to(self.device)
            with torch.no_grad():
                with autocast():
                    preds = self.model(sent_id, mask)
                    loss = self.criterion(preds, labels)
                val_loss_counter.update(loss, len(labels))
        return val_loss_counter.avg()


def loss_fn(y_true, y_pred):
    return nn.functional.cross_entropy(y_true, y_pred.to(torch.long))


best_scores = []
for model_num in range(5):
    print(f"{bb_}{w_}  Model#{model_num + 1}  {sr_}")

    tokenizer = AutoTokenizer.from_pretrained(Config.pretrained_model_path)
    config = AutoConfig.from_pretrained(Config.pretrained_model_path)
    config.update({
        "hidden_dropout_prob": 0.0,
        "layer_norm_eps": 1e-7,
        "output_hidden_states": True
    })

    train_dl = make_dataloader(kfold_df[kfold_df.fold != model_num], tokenizer)
    val_dl = make_dataloader(kfold_df[kfold_df.fold == model_num], tokenizer, is_train=False)
    transformer = AutoModel.from_pretrained(Config.pretrained_model_path, config=config)

    model = CLRPModel(transformer, config)

    model = model.to(Config.device)
    optimizer = create_optimizer(model)
    scaler = GradScaler()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=Config.epochs * len(train_dl),
        num_warmup_steps=len(train_dl) * Config.epochs * 0.11)

    criterion = loss_fn

    trainer = Trainer(train_dl, val_dl, model, optimizer, scheduler, scaler, criterion, model_num)
    record_info, best_val_loss = trainer.run()
    best_scores.append(best_val_loss)

    steps, train_losses = list(zip(*record_info["train_loss"]))
    plt.plot(steps, train_losses, label="train_loss")
    steps, val_losses = list(zip(*record_info["val_loss"]))
    plt.plot(steps, val_losses, label="val_loss")
    plt.legend()
    plt.show()

print("Best val losses:", best_scores)
print("Avg val loss:", np.array(best_scores).mean())

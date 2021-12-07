import time
import warnings

warnings.filterwarnings("ignore")

import torch
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

class Trainer:
    def __init__(self, train_dl, val_dl, model, optimizer, scheduler, scaler, criterion, model_num, Config):
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
        self.Config = Config

    def run(self):
        record_info = {
            "train_loss": [],
            "val_loss": [],
        }

        best_val_loss = float("inf")
        evaluation_scheduler = EvaluationScheduler(self.Config.eval_schedule)
        train_loss_counter = AvgCounter()
        step = 0

        for epoch in range(self.Config.epochs):
            print(f"{r_}Epoch: {epoch + 1}/{self.Config.epochs}{sr_}")
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
                        torch.save(self.model, self.Config.model_dir / f"best_model_{self.model_num}.pt")

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

        if (batch_step + 1) % self.Config.gradient_accumulation or batch_step + 1 == self.total_batch_steps:
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
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from transformers import AutoModelForMaskedLM, AutoTokenizer, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
import re


class Config:
    model_name = "cl-tohoku/bert-base-japanese-v2"
    pretrained_model_path = "./bert-base-japanese"
    output_hidden_states = True
    epochs = 3
    batch_size = 16
    device = "cuda"
    seed = 42
    max_len = 256
    train_dir = "./train_val_split"
    item = ["story", "title", "keyword"]
    item_num = 0  # 0ならstory(あらすじ),1ならtitle(題名),2ならkeyword(タグ)
    dataset_dir = "../input/nishika-narou"


train_data = pd.read_csv(Config.train_dir + "/kfold_2021_07.csv")
test_data = pd.read_csv(Config.dataset_dir + "/test.csv")

train_data = train_data.drop(columns=["fold"])

data = pd.concat([train_data, test_data])
data["excerpt"] = data[Config.item[Config.item_num]]
data["excerpt"] = data["excerpt"].apply(lambda x: x.replace("\n", ""))


def remove_url(sentence):
    ret = re.sub(r"(http?|ftp)(:\/\/[-_\.!~*\"()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", str(sentence))
    return ret


data.excerpt = data.excerpt.map(remove_url)
text = "\n".join(data.excerpt.tolist())
with open("text.txt", "w") as f:
    f.write(text)

model = AutoModelForMaskedLM.from_pretrained(Config.model_name)
tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
tokenizer.save_pretrained(Config.pretrained_model_path)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",
    block_size=256)
valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",
    block_size=256)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir="./nishika_narou_chk",
    overwrite_output_dir=True,
    num_train_epochs=Config.epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    save_total_limit=2,
    eval_steps=400,
    metric_for_best_model="eval_loss",
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

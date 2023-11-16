import pandas as pd
from datasets import load_dataset , Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer   ,DataCollatorForSeq2Seq , pipeline , TrainingArguments, Trainer
import torch

def get_feature(batch):
  encodings = tokenizer(batch['judgement'], text_target=batch['summary'],
                        max_length=1024, truncation=True)

  encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

  return encodings

device = 'gpu'
model_ckpt = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

data = pd.read_csv('Train_data.csv')
subset_df = data.sample(n=500)
subset_df.reset_index(drop=True, inplace=True)
test =  pd.read_csv('Test_data.csv')
huggingface_dataset = Dataset.from_pandas(subset_df)
test_dataset =  Dataset.from_pandas(test)


#
data_pt = huggingface_dataset.map(get_feature, batched=True)
test_pt =  test_dataset.map(get_feature, batched = True)

columns = ['input_ids', 'labels', 'attention_mask']
data_pt.set_format(type='torch', columns=columns)
test_pt.set_format(type='torch', columns=columns)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir = 'bart_Legal',
    num_train_epochs= 2,
    warmup_steps = 500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay = 0.01,
    logging_steps = 10,
    evaluation_strategy = 'steps',
    eval_steps=500,
    save_steps=1e6,
    gradient_accumulation_steps=16
)

trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator,
                  train_dataset = data_pt, eval_dataset = test_pt)

trainer.train()

trainer.save_model('bart_Legal_india_model')







from LoadData import CustomDataset
import json
import torch
from torch.utils.data import Dataset, DataLoader


# Setting up the device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")

train_dataset = CustomDataset('train.jsonl')
val_dataset = CustomDataset('valid.jsonl')

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

from transformers import LongformerForSequenceClassification

model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2)

model.to(device)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # where to save the model
    num_train_epochs=5,              # number of training epochs
    per_device_train_batch_size=2,   # batch size for training
    per_device_eval_batch_size=2,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch"      # evaluate at the end of each epoch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

trainer.evaluate()

model.save_pretrained('./saved_model')

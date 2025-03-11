#!/usr/bin/env python
# coding: utf-8

# In[16]:


import transformers as T
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn
import torch.nn.utils.rnn
from torchmetrics import SpearmanCorrCoef, Accuracy, F1Score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[17]:


# 有些中文的標點符號在tokenizer編碼以後會變成[UNK]，所以將其換成英文標點
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]


# In[18]:


class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation","test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # 把中文標點替換掉
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")


# In[19]:


# Define the hyperparameters
lr = 3e-5
epochs = 3
train_batch_size = 8
validation_batch_size = 8


# In[20]:


data_sample1 = SemevalDataset(split="train")


# In[21]:


print(len(data_sample1))


# In[22]:


# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.

def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.
    premises = [item['premise'] for item in batch]
    hypotheses = [item['hypothesis'] for item in batch]
    relatedness_scores = torch.tensor([item['relatedness_score'] for item in batch], dtype=torch.float32).to(device)
    entailment_judgments = torch.tensor([item['entailment_judgment'] for item in batch], dtype=torch.long).to(device)

    inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt").to(device)
    return inputs, relatedness_scores, entailment_judgments

# TODO1-2: Define your DataLoader
dl_train =   DataLoader(SemevalDataset(split="train"), batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)# Write your code here
dl_validation = DataLoader(SemevalDataset(split="validation"), batch_size=validation_batch_size, collate_fn=collate_fn)# Write your code here
dl_test = DataLoader(SemevalDataset(split="test"), batch_size=validation_batch_size, collate_fn=collate_fn)# Write your code here


# In[30]:


# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Write your code here
        # Define what modules you will use in the model
        self.bert = T.BertModel.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco", cache_dir="./cache/")
        self.dropout = torch.nn.Dropout(0.3)
        self.fc_relatedness = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.fc_entailment = torch.nn.Linear(self.bert.config.hidden_size, 3)


    def forward(self, **kwargs):
        # Write your code here
        # Forward pass
        return self.bert(**kwargs)
        


# In[29]:


model = MultiLabelModel().to(device)
tokenizer = T.BertTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco", cache_dir="./cache/")


# In[25]:


# inputs, labels1, labels2 = next(iter(dl_train))

# print("input_text:")
# print("input_ids:", inputs['input_ids'])
# print("token_type_ids:", inputs['token_type_ids'])
# print("attention_mask:", inputs['attention_mask'])

# print("\nlabels1 (regression):", labels1)
# print("\nlabels2 (classification):", labels2)


# In[26]:


# TODO3: Define your optimizer and loss function
import torch.optim as optim
# TODO3-1: Define your Optimizer
optimizer =optim.AdamW(model.parameters(), lr=lr)  # Write your code here

# TODO3-2: Define your loss functions (you should have two)
# Write your code here
loss_fn_relatedness = torch.nn.MSELoss()  # For the regression task
loss_fn_entailment = torch.nn.CrossEntropyLoss()  # For the classification task
# scoring functions
spc = SpearmanCorrCoef()
acc = Accuracy(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3, average='macro')


# In[27]:


import os

for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop
    # Write your code here
    # train your model
    # clear gradient
    # forward pass
    # compute loss
    # back-propagation
    # model optimization
    for batch in pbar:
        optimizer.zero_grad()
        inputs, relatedness_scores, entailment_judgments = batch
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output
        logits_relatedness = model.fc_relatedness(pooled_output).squeeze(-1)
        logits_entailment = model.fc_entailment(pooled_output)

        loss_relatedness = loss_fn_relatedness(logits_relatedness, relatedness_scores)
        loss_entailment = loss_fn_entailment(logits_entailment, entailment_judgments)
        loss = loss_relatedness + loss_entailment

        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # TODO5: Write the evaluation loop
    # Write your code here
    # Evaluate your model
    # Output all the evaluation scores (SpearmanCorrCoef, Accuracy, F1Score)
    total_loss = 0
    total_relatedness = []
    total_relatedness_pred = []
    total_entailment = []
    total_entailment_pred = []

    with torch.no_grad():
        for batch in pbar:
            inputs, relatedness_scores, entailment_judgments = batch
            outputs = model(**inputs)
            pooled_output = outputs.pooler_output
            logits_relatedness = model.fc_relatedness(pooled_output).squeeze(-1)
            logits_entailment = model.fc_entailment(pooled_output)

            loss_relatedness = loss_fn_relatedness(logits_relatedness, relatedness_scores)
            loss_entailment = loss_fn_entailment(logits_entailment, entailment_judgments)
            loss = loss_relatedness + loss_entailment

            total_loss += loss.item()
            total_relatedness.extend(relatedness_scores.cpu().numpy())
            total_relatedness_pred.extend(logits_relatedness.cpu().numpy())
            total_entailment.extend(entailment_judgments.cpu().numpy())
            total_entailment_pred.extend(torch.argmax(logits_entailment, dim=1).cpu().numpy())

    avg_loss = total_loss / len(dl_train)
    spearman_corr = spc(torch.tensor(total_relatedness_pred), torch.tensor(total_relatedness))
    accuracy = acc(torch.tensor(total_entailment_pred), torch.tensor(total_entailment))
    f1_score = f1(torch.tensor(total_entailment_pred), torch.tensor(total_entailment))

    print(f"Validation Loss: {avg_loss}")
    print(f"Spearman Correlation: {spearman_corr}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')


    torch.save(model, f'./saved_models/ep.ckpt')


# In[28]:


model = torch.load('./saved_models/ep.ckpt')
model.eval()
# TODO5: Write the evaluation loop
# Write your code here
# Evaluate your model
# Output all the evaluation scores (SpearmanCorrCoef, Accuracy, F1Score)
total_loss = 0
total_relatedness = []
total_relatedness_pred = []
total_entailment = []
total_entailment_pred = []
pbar = tqdm(dl_test)

with torch.no_grad():
    for batch in pbar:
        inputs, relatedness_scores, entailment_judgments = batch
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output
        logits_relatedness = model.fc_relatedness(pooled_output).squeeze(-1)
        logits_entailment = model.fc_entailment(pooled_output)

        loss_relatedness = loss_fn_relatedness(logits_relatedness, relatedness_scores)
        loss_entailment = loss_fn_entailment(logits_entailment, entailment_judgments)
        loss = loss_relatedness + loss_entailment

        total_loss += loss.item()
        total_relatedness.extend(relatedness_scores.cpu().numpy())
        total_relatedness_pred.extend(logits_relatedness.cpu().numpy())
        total_entailment.extend(entailment_judgments.cpu().numpy())
        total_entailment_pred.extend(torch.argmax(logits_entailment, dim=1).cpu().numpy())

avg_loss = total_loss / len(dl_test)
spearman_corr = spc(torch.tensor(total_relatedness_pred), torch.tensor(total_relatedness))
accuracy = acc(torch.tensor(total_entailment_pred), torch.tensor(total_entailment))
f1_score = f1(torch.tensor(total_entailment_pred), torch.tensor(total_entailment))

print(f"Validation Loss: {avg_loss}")
print(f"Spearman Correlation: {spearman_corr}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")


# For test set predictions, you can write perform evaluation simlar to #TODO5.

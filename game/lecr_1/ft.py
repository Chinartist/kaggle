# %% [markdown]
# # Finetuning SentenceTransformer

# %%

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
seed_everything(42)



def read_data():
    DATA_PATH = "/mnt/hdd1/wangjingqi/dataset/lecr/"
    topics = pd.read_csv(DATA_PATH + "topics.csv")
    content = pd.read_csv(DATA_PATH + "content.csv")
    correlations = pd.read_csv(DATA_PATH + "correlations.csv")
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    topics['description'].fillna("", inplace = True)
    content['description'].fillna("", inplace = True)
    content['text'].fillna("", inplace = True)
    topics["Ti"] = topics["title"]
    content["Ti"] = content["title"]
    
    topics["TiDe"] = topics["title"]+" "+topics["description"]
    content["TiDe"] = content["title"]+" "+content["description"]

    topics["TiDeTe"] = topics["title"]+" "+topics["description"]
    content["TiDeTe"] = content["title"]+" "+content["description"]+" "+content["text"]
    return topics, content, correlations
uns_key = "TiDeTe"
print("uns_key:", uns_key)
topics, content, correlations = read_data()
topics.rename(columns=lambda x: "topic_" + x, inplace=True)
content.rename(columns=lambda x: "content_" + x, inplace=True)

correlations["content_id"] = correlations["content_ids"].str.split(" ")
corr = correlations.explode("content_id").drop(columns=["content_ids"])

corr = corr.merge(topics, how="left", on="topic_id")
corr = corr.merge(content, how="left", on="content_id")
corr.head()

corr["set"] = corr[[f"topic_{uns_key}", f"content_{uns_key}"]].values.tolist()
train_df = pd.DataFrame(corr["set"])

dataset = Dataset.from_pandas(train_df)

train_examples = []
train_data = dataset["set"]
n_examples = dataset.num_rows

for i in range(n_examples):
    example = train_data[i]
    if example[0] == None: #remove None
        print(example)
        continue        
    train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))

model_name = [ 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'sentence-transformers/all-MiniLM-L6-v2'][0]
model = SentenceTransformer(model_name, device=3)
model._modules['0'].max_seq_length=50


num_epochs = 30
if model_name == 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2':
    checkpoint_path="/mnt/hdd1/wangjingqi/ck/lecr/ft/pmmb2"+f"_{uns_key}"
    batch_size = 560
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,optimizer_params={'lr': 2.5e-4},scheduler="WarmupLinear",
            warmup_steps=warmup_steps,checkpoint_path=checkpoint_path,
                checkpoint_save_steps=len(train_dataloader))
elif model_name == 'sentence-transformers/all-MiniLM-L6-v2':
    checkpoint_path="/mnt/hdd1/wangjingqi/ck/lecr/ft/aml2"+f"_{uns_key}"
    batch_size = 1024#1e-3
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,optimizer_params={'lr': 1e-3},scheduler="WarmupLinear",
            warmup_steps=warmup_steps,checkpoint_path=checkpoint_path,
                checkpoint_save_steps=len(train_dataloader))
    

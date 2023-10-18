# %% [markdown]
# # Finetuning SentenceTransformer

# %%
import cupy as cp
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from tqdm import tqdm
seed_everything(42)
import os
from transformers import AutoTokenizer, AutoModel
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
def read_data():
    content = pd.read_csv("/mnt/hdd1/wangjingqi/dataset/lecr/content.csv")
    topics = pd.read_csv("/mnt/hdd1/wangjingqi/dataset/lecr/topics.csv")
    correlations = pd.read_csv("/mnt/hdd1/wangjingqi/dataset/lecr/correlations.csv")
    
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    topics['description'].fillna("", inplace = True)
    content['description'].fillna("", inplace = True)
    content['text'].fillna("", inplace = True)

    topics["input"] = topics["title"]+" "+topics["description"]
    content["input"] = content["title"]+" "+content["description"]+" "+content["text"]

    topics['length'] = topics["input"].apply(lambda x: len(x))
    content['length'] = content["input"].apply(lambda x: len(x))
    
    topics.sort_values('length', inplace = True)
    content.sort_values('length', inplace = True)
    # Drop cols
    topics.drop(['title','description', 'channel', 'category', 'level', 'has_content', 'length'], axis = 1, inplace = True)
    content.drop(['title','description', 'kind',  'text', 'copyright_holder', 'license', 'length'], axis = 1, inplace = True)
    # Reset index
    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    print(f"correlations.shape: {correlations.shape}")
    return topics, content, correlations
topics, content, correlations = read_data()
train_df = pd.read_csv(f'/mnt/hdd1/wangjingqi/dataset/lecr/train_32.csv').reset_index(drop=True)
train_df["input1"].fillna("", inplace = True)
train_df["input2"].fillna("", inplace = True)
# train_df = train_df.sample(1000).reset_index(drop=True)

from sklearn.model_selection import StratifiedGroupKFold
kfold = StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state =42)
for num, (train_index, val_index) in enumerate(kfold.split(train_df, train_df['target'], train_df['topics_ids'])):
        train_df.loc[val_index, 'fold'] = int(num)
        topics.loc[topics['id'].isin(train_df.loc[val_index, 'topics_ids']), 'fold'] = int(num)
        # content.loc[content['id'].isin(train_df.loc[val_index, 'content_ids']), 'fold'] = int(num)
        correlations.loc[correlations['topic_id'].isin(train_df.loc[val_index, 'topics_ids']), 'fold'] = int(num)
def get_examples(train_df):
    train_df["set"] = train_df[["input1", "input2"]].values.tolist()
    dataset  =Dataset.from_pandas(train_df[['set','target']])
    n_examples = dataset.num_rows
    train_examples = []
    for i in tqdm(range(n_examples)):
        example = dataset[i]
        if example["set"][0] == None: #remove None
            print(example)
            continue        
        train_examples.append(InputExample(texts=[str(example["set"][0]), str(example["set"][1])],label=example["target"]))
    print(len(train_examples))
    return train_examples


from sentence_transformers.evaluation import SentenceEvaluator
import logging
import torch

import os
import gc
class RecallEvaluator(SentenceEvaluator):
   
    def __init__(self, data,checkpoint_path=None, model_name=None):
        self.data = data
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.best_score = 0
        self.best_epoch = 0
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info(f"RecallEvaluator: Evaluating the model {self.checkpoint_path}"  + out_txt)
        print(f"RecallEvaluator: Evaluating the model {self.checkpoint_path}"  + out_txt)
        scores = self.compute_metrices(model)
        logging.info(f"RecallEvaluator:{scores}")
        print(f"RecallEvaluator:{scores}")
        if epoch is not None:
            model.save(os.path.join(self.checkpoint_path, f"{epoch}-{steps}"))
            if scores['recall@50'] > self.best_score:
                self.best_score = scores['recall@50']
                self.best_epoch = epoch
                logging.info(f"RecallEvaluator: Best score is {self.best_score} at epoch {self.best_epoch}")
                print(f"RecallEvaluator: Best score is {self.best_score} at epoch {self.best_epoch}")
                model.save(os.path.join(self.checkpoint_path, f"best"))
        return scores['recall@50']
    @torch.no_grad()
    def compute_metrices(self, model):
        model.eval()
        topics = self.data["topics"]
        content = self.data["content"]
        correlations = self.data["correlations"]
       # Create topics dataset
        topics_dataset = LECRDataset(topics,"input")
        # Create content dataset
        content_dataset = LECRDataset(content,"input")
        
        collate_fn= collator(self.model_name,model._modules['0'].max_seq_length)
        # Create topics and content dataloaders
        topics_loader = DataLoader(topics_dataset,batch_size = 256, shuffle = False, num_workers=4, pin_memory=True,collate_fn =collate_fn,drop_last=False)
        content_loader = DataLoader(content_dataset,batch_size = 256, shuffle = False, num_workers= 4, pin_memory=True,collate_fn =collate_fn,drop_last=False)
        topics_preds = get_embeddings(topics_loader, model, model._target_device)

        content_preds = get_embeddings(content_loader, model, model._target_device)
        # Transfer predictions to gpu
        topics_preds_gpu = cp.array(topics_preds)
        content_preds_gpu = cp.array(content_preds)
        scores =self.get_nbs_scores(content_preds_gpu, topics_preds_gpu, topics, content, correlations)
        del topics_preds_gpu, content_preds_gpu
        torch.cuda.empty_cache()
        model.train()
        return scores
    def get_nbs_scores(self,content_preds_gpu, topics_preds_gpu, topics, content, correlations):
        
        scores = {}
        for top_n in [5,20,50,100]:
            neighbors_model = NearestNeighbors(n_neighbors = top_n, metric = 'cosine')
            neighbors_model.fit(content_preds_gpu)
            indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
            predictions = []
            for k in range(len(indices)):
                pred = indices[k]
                p = ' '.join([content.loc[ind, 'id'] for ind in pred.get()])
                predictions.append(p)
            topics['predictions'] = predictions

            corr = topics.merge(correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
            scores[f'recall@{top_n}'] = self.get_pos_score(corr['content_ids'], corr['predictions'])
        return scores
    def get_pos_score(self,y_true, y_pred):
        y_true = y_true.apply(lambda x: set(x.split()))
        y_pred = y_pred.apply(lambda x: set(x.split()))
        int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
        return round(np.mean(int_true), 5)
class LECRDataset(torch.utils.data.Dataset):
    def __init__(self,df,key):
        self.inputs = df[key].values
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        sample = self.inputs[idx]
        return sample

class collator():
    def __init__(self,pretrained_path,max_len=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.max_len = max_len
    def __call__(self, data):
        inputs = self.tokenize(list(data))
        return inputs
    def tokenize(self,texts):
            return self.tokenizer(
                texts,padding='longest',max_length=self.max_len,truncation=True,return_tensors="pt",return_token_type_ids=False)

def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)["sentence_embedding"]
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds

model_name = [ 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'sentence-transformers/all-MiniLM-L6-v2'][0]
device = 1

model = SentenceTransformer(model_name_or_path=model_name, device=device)
model._modules['0'].max_seq_length=42
num_epochs = 24
fold=1
margin = 0.3
if model_name == 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2':
    batch_size = 420
    lr = 6e-5
    checkpoint_path=f"/mnt/hdd1/wangjingqi/ck/lecr/ft/pmmb2_{lr*1e5}_{num_epochs}_{batch_size}_{model._modules['0'].max_seq_length}_{margin}_{fold}"
    log_file =f"/mnt/hdd1/wangjingqi/lecr_1/log/pmmb2_{lr*1e5}_{num_epochs}_{batch_size}_{model._modules['0'].max_seq_length}_{margin}_{fold}.txt"
    with open(log_file, 'w') as f:
        f.write(log_file)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    train_examples = get_examples(train_df[train_df.fold != fold].reset_index(drop=True))
    eval_data = {"topics":topics[topics.fold == fold].reset_index(drop=True), "content":content, "correlations":correlations[correlations.fold == fold].reset_index(drop=True)}
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size,drop_last=True,num_workers=8,pin_memory=True)
    model.to(model._target_device)
    ev = RecallEvaluator(eval_data,checkpoint_path,model_name)
    ev(model, output_path=checkpoint_path, epoch=None, steps=None)
    model.train()
    train_loss = losses.ContrastiveLoss(model=model,margin=margin)
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data
    model.fit(train_objectives=[(train_dataloader, train_loss)],evaluator=RecallEvaluator(eval_data,checkpoint_path,model_name),
            epochs=num_epochs,optimizer_params={'lr': lr},scheduler="WarmupLinear",
            warmup_steps=warmup_steps)
        
    # elif model_name == 'sentence-transformers/all-MiniLM-L6-v2':
    #     batch_size = 1024#1e-3
    #     checkpoint_path=f"/mnt/hdd1/wangjingqi/ck/lecr/ft/aml2_{num_epochs}_{batch_size}_{fold}"
        
    #     train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    #     train_loss = losses.ContrastiveLoss(model=model)
    #     warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data
    #     model.fit(train_objectives=[(train_dataloader, train_loss)],
    #             epochs=num_epochs,optimizer_params={'lr': 1e-3},scheduler="WarmupLinear",
    #             warmup_steps=warmup_steps,checkpoint_path=checkpoint_path,
    #                 checkpoint_save_steps=len(train_dataloader))
  # =========================================================================================
# =========================================================================================
# Libraries
# =========================================================================================

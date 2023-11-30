import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn import metrics
import transformers
import pandas as pd
import numpy as np
import time
import gc
import traceback
import datetime

from dataset import TrainDataset
from model import Model




def train_model(model, optimizer, scheduler, loss_function, epochs,train_data_loader,clip_value=2):
    try:
        model.train()
        for epoch in range(epochs):
            best_loss = []
            for step,batch in enumerate(train_data_loader):
                batch_inputs, batch_masks, batch_labels = batch['ids'], batch['mask'], batch['targets']
                batch_token_type_ids = batch['token_type_ids']
                model.zero_grad()
                outputs = model(batch_inputs, batch_masks, batch_token_type_ids)
                loss = loss_function(outputs.squeeze(),batch_labels.squeeze())
                best_loss.append(loss)
                loss.backward()
                clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                scheduler.step()
            loss_avg = np.mean(best_loss)
            print("Epoch : {0} , loss : {1}".format(epoch,loss_avg))
        return model
    
    except Exception as e:
        print("Error in train_model >>>>>>")
        traceback.print_exc()
        return None
        

def evaluate_model(model,loss_function,valid_data_loader):
    try:
        model.eval()
        valid_loss, valid_acc = [], []
        for step,batch in enumerate(valid_data_loader):
            batch_inputs, batch_masks, batch_labels = batch['ids'], batch['mask'], batch['targets']
            batch_token_type_ids = batch['token_type_ids']
            with torch.no_grad():
                outputs = model(batch_inputs, batch_masks, batch_token_type_ids)
            loss = loss_function(outputs, batch_labels)
            valid_loss.append(loss.item())
            acc = metrics.accuracy_score(outputs, batch_labels)
            valid_acc.append(acc)
        return valid_loss, valid_acc

    except Exception as e:
        print("Error in evaluate_model >>>>>>")
        traceback.print_exc()
        return None,None


def cross_validation(df,num_classes,k_fold):
    epochs = 10  # no of epochs 
    MAX_LENGTH = 64 # maxium length of embeddings (input to model)
    tokenizer_path = "bert-base-uncased"  # add tokenizer
    model_path = "bert-base-uncased" # add model path
    train_batch_size = 32  # training batch size
    learn_rate = 1e-5  # learning rate

    for fold in range(1,k_fold+1):
        print('>>>> Starting for fold : ',fold)
        df_train = df[df['kfold'] != fold].reset_index(drop=True)
        df_valid = df[df['kfold'] == fold].reset_index(drop=True)
        
        train_dataset = TrainDataset(
            texts = df_train['dis_symptoms'].values,
            targets = df_train['encoded_dis_name'].values,
            tokenizer_path = tokenizer_path,
            max_len = MAX_LENGTH
        )
        valid_dataset = TrainDataset(
            texts = df_valid['dis_symptoms'].values,
            targets = df_valid['encoded_dis_name'].values,
            tokenizer_path = tokenizer_path,
            max_len = MAX_LENGTH
        )
        train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=train_batch_size,shuffle=True)
        print('>>>> Train and Valid Dataset loaded')

        num_train_steps = len(train_data_loader) * epochs
        model = Model(model_path=model_path,num_classes=num_classes)
        print('>>>> Model Intialized')

        optimizer = torch.optim.AdamW(model.parameters(),lr=learn_rate,eps=1e-8)
        loss_function = nn.CrossEntropyLoss()
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )
        
        xx = time.time()
        print('>>>> Starting Model Train :- ',datetime.datetime.now())
        model = train_model(model, optimizer, scheduler, loss_function, epochs,train_data_loader)
        print('>>>> Model Train time taken :- ',time.time()-xx)
        if model is not None:
            valid_loss, valid_acc = evaluate_model(model,loss_function,valid_data_loader)
            print('>>>> Validation loss :- ',valid_loss)
            print('>>>> Validation accuracy :- ',valid_acc)   
        else:
            print("!!!!!!! Training Failed !!!!!!!")

        del model
        gc.collect()
        break

   

if __name__ == "__main__":
    sym = pd.read_csv("./data/final_dataset_all.csv")
    dis = pd.read_csv("./data/unique_diseases_info.csv")
    num_classes = len(list(dis['dis_name'].values))
    print('num_classes :- ',num_classes)
    k_fold = 5
    cross_validation(sym,num_classes,k_fold)
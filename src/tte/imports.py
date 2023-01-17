#!pip uninstall pandas
#!conda install numpy
#!pip install blis
#!pip install scipy
#!pip install lexnlp
#!pip install transformers==4.4.2
#!pip install torch
#!pip install sentencepiece==0.1.95
#!pip install sklearn
#!pip install nltk==3.2.5
#!pip install spacy==2.2.4
#!pip install sacremoses==0.0.43
#!pip install pandas==1.1.5
#!pip install matplotlib
#!pip install seaborn
#!pip install pickle
#!pip install pytorch==1.11.0 
#!pip install torchvision==0.12.0 
#!pip install torchaudio==0.11.0 
#!pip install cudatoolkit=11.3     



import torch  
import torch.cuda
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import XLMRobertaTokenizer              
from transformers import XLMRobertaForSequenceClassification
from transformers import AdamW                            
from transformers import get_linear_schedule_with_warmup
import sentencepiece

#sklearn for evaluation
from sklearn.metrics import classification_report        
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid         
from sklearn.model_selection import ParameterSampler      
from sklearn.utils.fixes import loguniform

#nlp preprocessing
from nltk import ngrams        
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords                          
from spacy.pipeline import SentenceSegmenter
from spacy.lang.es import Spanish
from spacy.pipeline import Sentencizer
from sacremoses import MosesTokenizer, MosesDetokenizer


#utilities
import pandas as pd
import nltk as nean
import glob, os
import time
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



   
def load_text_corpus(path):
  text_data=""
  print("Path:",glob.glob(path))
  for file in glob.glob(path+"*.txt"):
      print("Documento:",file)
      with open(file) as f:
        temp_data = f.read()
        text_data=text_data+" "+temp_data
  print("Palabras:",len(text_data))
  return text_data

def preprocess(text):
  sentencizer = Sentencizer()
  nlp = Spanish()
  nlp.add_pipe(sentencizer)
  doc = nlp(text)
  #tokenize
  sentence_list=[]
  mt = MosesTokenizer(lang='es')
  for s in doc.sents:
    tokenized_text = mt.tokenize(s, return_str=True)
    sentence_list.append((tokenized_text.split(), s))     #append tuple of tokens and original senteence
  return sentence_list

def create_training_data(sentence_list, df_terms, n):
  training_data = pd.DataFrame(columns=['n_gram', 'Context', 'Label'])
  md = MosesDetokenizer(lang='es')
  print("Frases:",len(sentence_list))
  count=0
  for sen in sentence_list:
    count+=1
    if count%100==0:print(count)
    s=sen[0]  
    for i in range(1,n+1):
      n_grams = ngrams(s, i)
      for n_gram in n_grams: 
        n_gram=md.detokenize(n_gram) 
        context=str(sen[1]).strip()
        if n_gram.lower() in df_terms.values:
          termtype="/"
          training_data = training_data.append({'n_gram': n_gram, 'Context': context, 'Label': 1}, ignore_index=True)
        else:
          training_data = training_data.append({'n_gram': n_gram, 'Context': context, 'Label': 0}, ignore_index=True)
  return training_data

def undersample(train_data):
  print("Before")
  print(train_data.Label.value_counts())
  count_class_0, count_class_1 = train_data.Label.value_counts()
  df_class_0 = train_data[train_data['Label'] == 0]
  df_class_1 = train_data[train_data['Label'] == 1]
  df_class_0_under = df_class_0.sample(count_class_1)
  df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
  print("After")
  print(df_test_under.Label.value_counts())
  return df_test_under

def preprocesarDatos(ruta):
    print("Comienza el procesamiento de datos")
    device = torch.device('cuda')
    print('Connected to GPU:', torch.cuda.get_device_name(0))
    ruta_ingresada_train = ruta+"/train/"
    ruta_ingresada_valid = ruta+"/valid/"
    ruta_ingresada_eval = ruta+"/eval/"
    corp_text_train=load_text_corpus(ruta_ingresada_train) # load test
    corp_text_valid=load_text_corpus(ruta_ingresada_valid)
    corp_text_eval=load_text_corpus(ruta_ingresada_eval)
    corp_s_list_train=preprocess(corp_text_train)   
    corp_s_list_valid=preprocess(corp_text_valid)  
    corp_s_list_eval=preprocess(corp_text_eval)        
    process_t=pd.read_csv(ruta+"/trainAnotated/traintokens.csv", delimiter="\t", names=["Term","Label"])   
    process_v=pd.read_csv(ruta+"/validAnotated/validtokens.csv", delimiter="\t", names=["Term","Label"])
    process_e=pd.read_csv(ruta+"/evalAnotated/evaltokens.csv", delimiter="\t", names=["Term","Label"])                                    
    labels=["Random", "Term"]
    train_data_corp=create_training_data(corp_s_list_train, process_t, 6)   
    valid_data_corp=create_training_data(corp_s_list_valid, process_v, 6)  
    eval_data_corp=create_training_data(corp_s_list_eval, process_e, 6)  
    train_data_corp=undersample(train_data_corp)
    valid_data_corp=undersample(valid_data_corp)
    eval_data_corp=undersample(eval_data_corp)
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    max_len=64
    input_ids_train, attn_masks_train, labels_all_train = tokenizer_xlm(train_data_corp, max_len,xlmr_tokenizer)
    input_ids_valid, attn_masks_valid, labels_all_valid = tokenizer_xlm(valid_data_corp, max_len,xlmr_tokenizer)
    input_ids_eval, attn_masks_eval, labels_all_eval = tokenizer_xlm(eval_data_corp, max_len,xlmr_tokenizer)
    train_dataset = TensorDataset(input_ids_train, attn_masks_train, labels_all_train)
    valid_dataset = TensorDataset(input_ids_valid, attn_masks_valid, labels_all_valid)
    eval_dataset = TensorDataset(input_ids_eval, attn_masks_eval, labels_all_eval)
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size) #random sampling
    valid_dataloader = DataLoader(valid_dataset, sampler = SequentialSampler(valid_dataset),batch_size = batch_size ) #sequential sampling
    eval_dataloader = DataLoader(eval_dataset, sampler = SequentialSampler(eval_dataset),batch_size = batch_size )
    print("Datos preprocesados")
    print("---------------------------")
    #return train_dataloader,valid_data_corp,process_v,xlmr_tokenizer,device,labels,valid_dataloader,corp_text_train
    return train_dataloader,eval_data_corp,process_e,xlmr_tokenizer,device,labels,valid_dataloader,corp_text_train,valid_data_corp,process_v


def tokenizer_xlm(data, max_len,xlmr_tokenizer):
  labels_ = []
  input_ids_ = []
  attn_masks_ = []
  for index, row in data.iterrows():
      sentence = row['n_gram']+". "+row["Context"]
      encoded_dict = xlmr_tokenizer.encode_plus(sentence,
                                                max_length=max_len, 
                                                padding='max_length',
                                                truncation=True, 
                                                return_tensors='pt')
      input_ids_.append(encoded_dict['input_ids'])
      attn_masks_.append(encoded_dict['attention_mask'])
      labels_.append(row['Label'])
  input_ids_ = torch.cat(input_ids_, dim=0)
  attn_masks_ = torch.cat(attn_masks_, dim=0)
  labels_ = torch.tensor(labels_)
  print('Encoder finished. {:,} examples.'.format(len(labels_)))
  return input_ids_, attn_masks_, labels_






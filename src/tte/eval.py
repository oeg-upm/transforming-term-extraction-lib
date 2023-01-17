import sys
from imports import *
from model import *



def extract_terms(validation_df, xlmr_model,xlmr_tokenizer,max_len,device,labels): 
  term_list=[]
  xlmr_model.eval()
  for index, row in validation_df.iterrows():
    sentence = row['n_gram']+". "+row["Context"]
    label=validation_df["Label"]
    encoded_dict = xlmr_tokenizer.encode_plus(sentence, 
                                                  max_length=max_len, 
                                                  padding='max_length',
                                                  truncation=True, 
                                                  return_tensors='pt') 
    input_id=encoded_dict['input_ids'].to(device)
    attn_mask=encoded_dict['attention_mask'].to(device)
    label=torch.tensor(0).to(device)
    with torch.no_grad():                
      output = xlmr_model(input_id,token_type_ids=None,attention_mask=attn_mask,labels=label)
      loss=output.loss
      logits=output.logits   
    logits = logits.detach().cpu().numpy()
    pred=labels[logits[0].argmax(axis=0)]
    if pred=="Term":
      term_list.append(row['n_gram'])
   
  return set(term_list)

def computeTermEvalMetrics(extracted_terms, gold_df):
  extracted_terms = set([item.lower() for item in extracted_terms])
  gold_set=set(gold_df)
  true_pos=extracted_terms.intersection(gold_set)
  recall=len(true_pos)/len(gold_set)
  if(len(true_pos)==0):
      print("Precisión muy baja")
      precision=0
      f1=0
  else:
      precision=len(true_pos)/len(extracted_terms)
      f1=2*(precision*recall)/(precision+recall)
  print("Intersection",len(true_pos))
  print("Gold",len(gold_set))
  print("Extracted",len(extracted_terms))
  print("Recall:", recall)
  print("Precision:", precision)
  print("F1:", f1)
  
  
def evaluar(evaldata,xlmr_model,process_e,xlmr_tokenizer,device):
    print("Comienza la evaluación:")
    labels=["Random","Term"]
    max_len=64
    extracted_terms_wind_en=extract_terms(evaldata, xlmr_model,xlmr_tokenizer,max_len,device,labels)
    computeTermEvalMetrics(extracted_terms_wind_en, process_e["Term"])
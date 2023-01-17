import sys
from imports import *
labels=["Random", "Term"]

def create_model(lr, eps, train_dataloader, epochs, device):
  xlmr_model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
  desc = xlmr_model.to(device)
  print('Connected to GPU:', torch.cuda.get_device_name(0))
  optimizer = AdamW(xlmr_model.parameters(),
                  lr = lr,   
                  eps = eps       
                )
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,   
                                            num_training_steps = total_steps)
  return xlmr_model, optimizer, scheduler


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  

def validate(validation_df, xlmr_model, print_cm,xmlr_tokenizer,device,gold_set): 
  max_len=64 
  xlmr_model.eval()
  extracted_terms=extract_terms(validation_df, xlmr_model,xmlr_tokenizer,max_len,device)
  extracted_terms_valid = set([item.lower() for item in extracted_terms])
  gold_set=set(gold_set)
  true_pos=extracted_terms_valid.intersection(gold_set)
  recall=len(true_pos)/len(gold_set)
  if(len(true_pos)==0):
      print("Precisión muy baja")
      precision=0
      f1=0
  else:
      precision=len(true_pos)/len(extracted_terms_valid)
      f1=2*(precision*recall)/(precision+recall)
  return recall, precision, f1,labels

def extract_terms(validation_df, xlmr_model,xlmr_tokenizer,max_len,device): 
  labels=["Random","Term"]
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
      output = xlmr_model(input_id,token_type_ids=None, attention_mask=attn_mask,labels=label)
      loss=output.loss
      logits=output.logits 
    logits = logits.detach().cpu().numpy()
    pred=labels[logits[0].argmax(axis=0)]
    if pred=="Term":
      term_list.append(row['n_gram'])
  return set(term_list) 
      
def train_model(epochs, xlmr_model, train_dataloader, validation_df, random_seed, optimizer, scheduler,device,xmlr_tokenizer,process_v):
  seed_val = random_seed
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)
  training_stats = []
  total_t0 = time.time()
  print('\033[1m'+"================ Model Training ================"+'\033[0m')
  for epoch_i in range(0, epochs):
      print("")
      print('\033[1m'+'======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs)+'\033[0m')
      t0 = time.time()
      total_train_loss = 0
      xlmr_model.train()
      for step, batch in enumerate(train_dataloader):
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)
          xlmr_model.zero_grad()        
          output = xlmr_model(b_input_ids, 
                              token_type_ids=None, 
                              attention_mask=b_input_mask, 
                              labels=b_labels)         
          loss=output.loss
          logits=output.logits
          total_train_loss += loss.item()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(xlmr_model.parameters(), 1.0)
          optimizer.step()
          scheduler.step()
      avg_train_loss = total_train_loss / len(train_dataloader)            
      training_time = format_time(time.time() - t0)
      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Training epoch took: {:}".format(training_time))
      print("  Evaluación de los resultados  ")
      if epoch_i==epochs-1:print_cm=True #Print out cm in final iteration
      else: print_cm=False
      recall, precision, f1,labels = validate( validation_df, xlmr_model, print_cm,xmlr_tokenizer,device,process_v["Term"])   
      training_stats.append(
          {
              'epoch': epoch_i + 1,
              'Training Loss': avg_train_loss,
              "precision": precision,
              "recall": recall,
              "f1": f1,
              'Training Time': training_time,
          })
      print("Precision", precision)
      print("Recall", recall)
      print("F1", f1)
  print("\n\nTraining complete!")
  print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
  return training_stats,labels

def procesarmodelo(train,valid_dataloader,device,valid_data_corp,xmlr_tokenizer,process_v):
    valid_data_corp=valid_data_corp
    lr=2e-7
    eps=1e-9
    epochs=3
    device = torch.device('cuda')
    xlmr_model, optimizer, scheduler = create_model(lr=lr,
                                                    eps=eps,
                                                    train_dataloader=train,
                                                    epochs=epochs,
                                                    device=device)
    training_stats=train_model(epochs=epochs,
                               xlmr_model=xlmr_model,
                               train_dataloader=train,
                               validation_df=valid_data_corp,
                               random_seed=42,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               device=device,
                               xmlr_tokenizer=xmlr_tokenizer,
                               process_v=process_v)
    print("Modelo procesado")
    print("----------------------------")
    return xlmr_model



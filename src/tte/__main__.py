import sys
from imports import *
from model import *
from eval import *

#insertar ruta de la carpeta que posee el corpus
ruta="D:/UPM/corpus"

torch.cuda.empty_cache()
#train,evaldata,process_v,xmlr_tokenizer,device,labels_all_eval,valid_dl,train_df=preprocesarDatos(ruta)
train,evaldata,process_e,xmlr_tokenizer,device,labels_all_eval,valid_dl,train_df,validdata,process_v=preprocesarDatos(ruta)
#xlmr_model=procesarmodelo(train,valid_dl,device,evaldata,xmlr_tokenizer,process_v)
xlmr_model=procesarmodelo(train,valid_dl,device,validdata,xmlr_tokenizer,process_v)
#evaluar(evaldata,xlmr_model,process_v,xmlr_tokenizer,device)
evaluar(evaldata,xlmr_model,process_e,xmlr_tokenizer,device)

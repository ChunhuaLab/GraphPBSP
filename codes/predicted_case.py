from torch.nn import Parameter
import torch.nn.functional as F
import  torch.nn  as nn
#from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold

import numpy as np 
import sys,os, gc
import csv, glob
import os.path as path
import torch
from models_v6 import *
import pandas as pd
import argparse, pickle
import random
import glob
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix,average_precision_score
from time import time
#from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import datetime
import itertools
import warnings

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.metrics._classification")

  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


def predicted(model_state_dict):
    model = PredModel(int(head),int(hidden),int(hidden)*2,int(hidden),note_feature,edge_features)

   
    model = model.to(device)
    model.load_state_dict(model_state_dict)  
    model.eval()
    prediction=[]
    true=[]
    pdbs=[]
    with torch.no_grad():
        #dev_loss = 0
        Recall, Specificity, Precision, Auc, Mcc, F1, Acc,Aupr,Auprc = 0,0,0,0,0,0,0,0,0
        #print(1)
        for sample in T_loader:
           
           pdbinf=sample[0]
           pdbf=pdbinf[:-1].upper()+'_'+pdbinf[-1].upper()
           #print(pdbf)
           protT5=protT5_dict[pdbf]
           
           s=sample[1]
           protT5=torch.tensor(protT5).unsqueeze(0)
           nodes=torch.FloatTensor(s[2]).unsqueeze(0)
           edges=torch.FloatTensor(s[3]).unsqueeze(0)
           neighb=torch.tensor(s[4]).unsqueeze(0)
           target=torch.tensor(s[1])
           
           protT5=protT5.to(device)
           edges = edges.to(device)
           nodes = nodes.to(device)
           neighb = neighb.to(device)
           target = target.to(device)
                     
           
           dev_outputs = model.get_pred(protT5,nodes,edges,neighb)               
           #print(dev_outputs.shape)
           #loss =  criterion(dev_outputs.squeeze(0), target.long())#
           #dev_loss += loss.item()             
          
           predict = torch.argmax(dev_outputs, dim=-1)           
           predict = predict.cpu().detach().numpy()
           actual = target.cpu().detach().numpy()
           #print(dev_outputs,predict, outputs,actual) 
           specificity, auc,recall,  precision,  mcc, f1, acc,aupr,auprc=metric(actual, predict,predict)    
           
           prediction.append(predict[0])
           true.append(actual)           
           pdbs.append(pdbf)
           
           Recall += recall
           Acc += acc
           Specificity += specificity
           Precision += precision
           Auc += auc
           Aupr += aupr
           Auprc += auprc
           Mcc += mcc
           F1 += f1      
        
    num=len(T_loader)
    #dev_loss=dev_loss/ num
    Recall =Recall/num
    Acc = Acc / num
    Specificity = Specificity / num
    Precision = Precision / num
    Auc = Auc / num
    Aupr = Aupr / num
    Auprc = Auprc / num
    Mcc = Mcc / num
    F1 = F1 /num       
    predicted_results=[true,prediction,pdbs]     
#    log = "predicted "
#    #log += ", {}: {:.5f}".format('CE', dev_loss)
#    
#    log += ", {}: {:.5f}".format('specificity', Specificity)
#    log += ", {}: {:.5f}".format('Auc', Auc)
#    log += ", {}: {:.5f}".format('Mcc', Mcc)
#    log += ", {}: {:.5f}".format('F1', F1)
#    log += ", {}: {:.5f}".format('Accuracy', Acc)
#    

#    print(log)   
    point=3
    #return dev_loss,np.round(Recall,point), np.round(Specificity,point), np.round(Precision,point), np.round(Auc,point), np.round(Mcc,point), np.round(F1,point), np.round(Acc,point),np.round(Aupr,point),predicted_results
    return np.round(Specificity,point),np.round(Auc,point),np.round(Recall,point),  np.round(Precision,point),  np.round(Mcc,point), np.round(F1,point), np.round(Acc,point),np.round(Aupr,point),np.round(Auprc,point),predicted_results
 
nearest_neighbors=25


#5nrm_A

casename=sys.argv[1].lower()+sys.argv[2]
Type=sys.argv[3]


with open(f"./example/features/protein_feature_results/protein_{Type}_{nearest_neighbors}_dict", "rb") as f:
    test_dict = pickle.load(f) 



#items_list1 = list(test_dict.items())
#test_dict = dict(items_list1[:5])

with open(f"./example/features/prostT5_{Type}_dict", "rb") as f:
    protT5_dict=pickle.load(f)



casedata=test_dict[casename]
T_loader=[[casename,casedata]]
print(f'{casename} exist!')

outppath='results'


head=6
hidden=32
n_fold=10
note_feature=24
edge_features=39
ratio=10
Epoch=360
learning=0.0001 

batch_size=2



modelpath=f'/models'

#outfile_result=f"./pred_result_{nearest_neighbors}/{testname}/head_{head}_hidden_{hidden}_batch_1_results"
outfile_result=f"{outppath}_{nearest_neighbors}"
os.system(f"mkdir {outfile_result}")

#T_loader= [[key,value] for key,value in test_dict.items()] 




 
    


#pred_out.write('Test has the best Specificity among all epoch for each fold\n')

pred_spe={}
pred_auc={}



for fold in range(n_fold):
    
    model_state_dict_auc = torch.load(f'./{modelpath}/model_fold_{fold}.pkl') # for inference, can refer to these few lines.
   
    
    d1,d2,d3,d4,d5,d6,d7,d8,d9,d_results=predicted(model_state_dict_auc)
    pred_auc[f'fold{fold}']=d_results

       

allpredd = []

#print(d_results[0])
for value in pred_auc.values():
    temp2=value[0]
    temp3=value[1]
    #print('temp3')
    allpredd.append(temp3) 
    
    
       


sumsd = [map(sum, zip(*sublist)) for sublist in zip(*allpredd)]
pred_meand = [[sum_value / len(allpredd) for sum_value in col_sum] for col_sum in sumsd]  
#pred_meand=[[1 if num > 0.5 else 0 for num in sublist] for sublist in pred_meand]

true_mean=temp0
#print(len(true_mean),len(pred_mean))




for num, (j,n,d) in enumerate(zip(temp0,pdbname,pred_meand)):

    df = pd.DataFrame()
   
   
    d=np.array(d)
    j=np.array(j)
    

    d_label = (d >0.5).astype(int)
         

    df[n+'_True']=j.astype(int)
   
    df['AUC_pred']=d_label.astype(int)
    excel_file = f'./{outfile_result}/{n}_pred_results.xlsx'
    df.to_excel(excel_file, index=False)  

    


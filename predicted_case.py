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
           
           prediction.append(predict[0])
           true.append(actual)           
           pdbs.append(pdbf)
           
     
        
    num=len(T_loader)
    #dev_loss=dev_loss/ num
    
    predicted_results=[true,prediction,pdbs]     

#    

#    print(log)   
    point=3
    #return dev_loss,np.round(Recall,point), np.round(Specificity,point), np.round(Precision,point), np.round(Auc,point), np.round(Mcc,point), np.round(F1,point), np.round(Acc,point),np.round(Aupr,point),predicted_results
    return predicted_results
 
nearest_neighbors=25
pdbname=sys.argv[1]

casename=pdbname.lower()+sys.argv[2]
Type=sys.argv[3]

with open(f"./example/features/protein_feature_results/protein_{Type}_{str(nearest_neighbors)}_dict", "rb") as f:
    test_dict = pickle.load(f) 



#items_list1 = list(test_dict.items())
#test_dict = dict(items_list1[:5])

with open("./example/features/prostT5_{Type}_dict", "rb") as f:
    protT5_dict=pickle.load(f)


#5nrm_A



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



modelpath=f'./models'

#outfile_result=f"./pred_result_{nearest_neighbors}/{testname}/head_{head}_hidden_{hidden}_batch_1_results"
outfile_result=f"{outppath}/{pdbname}"
os.system(f"mkdir {outfile_result}")



pred_spe={}
pred_auc={}



for fold in range(n_fold):
    
    model_state_dict_auc = torch.load(f'./{modelpath}/model_fold_{fold}.pkl') # for inference, can refer to these few lines.


       
    
    d_results=predicted(model_state_dict_auc)
    pred_auc[f'fold{fold}']=d_results

    

allpreds = []
allpredd = []
for value in pred_spe.values():
    
    temp0=value[0]
    temp1=value[1]
    pdbname= value[2]
    allpreds.append(temp1)
#print(d_results[0])
for value in pred_auc.values():
    temp2=value[0]
    temp3=value[1]
    #print('temp3')
    allpredd.append(temp3) 
    
    
       
sums = [map(sum, zip(*sublist)) for sublist in zip(*allpreds)]
pred_mean = [[sum_value / len(allpreds) for sum_value in col_sum] for col_sum in sums]  


sumsd = [map(sum, zip(*sublist)) for sublist in zip(*allpredd)]
pred_meand = [[sum_value / len(allpredd) for sum_value in col_sum] for col_sum in sumsd]  


true_mean=temp0
#print(len(true_mean),len(pred_mean))




for num, (i, j,n,d) in enumerate(zip(pred_mean, temp0,pdbname,pred_meand)):

    df = pd.DataFrame()
   
    i=np.array(i)
    d=np.array(d)
    j=np.array(j)
    
    i_label = (i >0.5).astype(int)
    d_label = (d >0.5).astype(int)
  
        


    df[n+'_True']=j.astype(int)
    df['SPE_pred']=i_label.astype(int)
    df['AUC_pred']=d_label.astype(int)
    excel_file = f'./{outfile_result}/{n}_pred_results.xlsx'
    df.to_excel(excel_file, index=False)  

    


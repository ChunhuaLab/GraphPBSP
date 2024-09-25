import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse,pickle
import numpy as np
from tqdm import tqdm
import gc,sys,os

Type=sys.argv[1]
fasta_file  ='./example/' + sys.argv[2]
output_path = './prostT5_'+Type
os.system(f'mkdir {output_path}')

def compute_prostT5():

    ID_list = []
    seq_list = []
    
    #print(nofile)
    for line in open(fasta_file):
          line=line.strip('\n').split('#')
          fileid=line[0]
          pdbid=fileid[0:4]
          pro=fileid[-1]      
          proseq=line[1]    
      
          ID_list.append(pdbid+'_'+pro)    
          seq_list.append(" ".join(list(proseq)))
    
   
    model_path = "../feature/prostT5/Rostlab/ProstT5"
    # Load the vocabulary and ProtT5-XL-UniRef50 Model
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()
    
    # Load the model into the GPU if avilabile and switch to inference mode
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.to(device)
    model = model.eval()
       
    batch_size = 1
    
    for i in tqdm(range(0, len(ID_list), batch_size)):
        if i + batch_size <= len(ID_list):
            batch_ID_list = ID_list[i:i + batch_size]
            batch_seq_list = seq_list[i:i + batch_size]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]
    
        # Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]
    
        # Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
        # Extracting sequences' features and load it into the CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
    
        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            np.save(output_path + "/" + batch_ID_list[seq_num], seq_emd)
            
               
def from_npy_dict():      	

    prostT5_dict={}
    for dir in os.listdir(output_path):
        name=(dir.split('.'))[0]
        print(name+' computed....')
        #generate dssp files
        prost=np.load(f'{output_path}/{dir}')
        
        prostT5_dict[name]=prost
    
    with open(f"./example/features/prostT5_{Type}_dict", "wb") as f:
        pickle.dump(prostT5_dict, f)
    #print(len(prostT5_dict))
    
compute_prostT5()    
from_npy_dict() 

os.system(f'mv {output_path} ./example/features ') 
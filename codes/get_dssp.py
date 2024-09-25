import os, pickle
import sys
import numpy as np
from Bio import pairwise2



def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        chain= lines[i][11]
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(8) # The last dim represents "Unknown" for missing residues
        if chain==pro_chain:
            seq += aa
            SS_vec[SS_type.find(SS)] = 1
            PHI = float(lines[i][103:109].strip())
            PSI = float(lines[i][109:115].strip())
            ACC = float(lines[i][34:38].strip())
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
            dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))
    #print((np.array(dssp_feature)).shape)
    #print(dssp_feature)
    return seq, dssp_feature

dataset= './example/'+sys.argv[1]
Type=sys.argv[2]   

      
pro_dssp={}
for i in open(dataset):

    temp=i.strip().split('#')
    fileid=temp[0]
    print(fileid)

    name=fileid[0:4]+'_'+ fileid[4]  

    pro_chain=fileid[-1]
    
    try:
      dssp_seq, dssp_matrix = process_dssp(f"./example/features/dssp_{Type}/{name}.dssp")
    except:
    	print(name,'dssp not excit')
    	
    pro_dssp[fileid]=[dssp_seq,dssp_matrix]
    
    
    if len(dssp_seq)<1:
    	 print(name,'dssp_seq<0')
#print(pro_dssp)   	 
with open(f"./example/features/pro_dssp_dict{Type}", "wb") as f:
    pickle.dump(pro_dssp, f)
#print(len(pro_dssp))	 
#print(pro_dssp)
#pro_chain ='N'
#seq='IILCPGCKGALMGCNMKACNCSIHVK'
#get_dssp('1WCO', seq)
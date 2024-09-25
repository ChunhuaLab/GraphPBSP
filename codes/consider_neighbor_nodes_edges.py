import torch
import sys,os
import pickle
from Bio import PDB,pairwise2
#from Bio import PDB, Align
#from Bio.Seq import Seq
#from Bio.SeqRecord import SeqRecord
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation
import numpy as np
import os.path as path
import logging
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning


warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=PDBConstructionWarning)

def split_data(protein_dict):
    test_protein_dict={}
    train_protein_dict={}
    testid=[]
    trainid=[]
    for line in open(testdata):              
        line=line.strip('\n').split('#')
        temp=line[0]
        testid.append(temp)
            
    for line in open(traindata):                        
        line=line.strip('\n').split('#')
        temp=line[0]
        trainid.append(temp)   
    print('The test  train  id number')        
    print(len(testid),len(trainid))
        
    for key in protein_dict.keys():       
        nameid=key.strip()        
        if nameid in trainid:
           train_protein_dict[key]=protein_dict[key]
        else:
           if nameid in testid:
              test_protein_dict[key]=protein_dict[key] 
           else:
              print(nameid)
           
    print('The test  train dict data number')        
    print(len(test_protein_dict),len(train_protein_dict))
    
    with open(f'{outpath}test_protein_neighbor{str(nearest_neighbors)}_dict', "wb") as f:
        pickle.dump(test_protein_dict,f)
        
    with open(f'{outpath}train_protein_neighbor{str(nearest_neighbors)}_dict', "wb") as f:
        pickle.dump(train_protein_dict,f)	

def three_to_one(three_letter_code):

    aa_dict = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V"
    }
    return aa_dict.get(three_letter_code, "X")
    
def match_dssp(dssp_seq, dssp, pdb_seq):
    #print(dssp)
     # The last dim represent "Unknown" for missing residues
    SS_vec = np.zeros(8)
    matched_dssp = []
    if dssp_seq != pdb_seq:
        
        alignments = pairwise2.align.globalxx(pdb_seq, dssp_seq)
        pdb_seq = alignments[0].seqA
        dsspseq = alignments[0].seqB
        padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))
        new_dssp = []
        for aa in dsspseq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))
            
        for i in range(len(pdb_seq)):
            if pdb_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])
    else:
        matched_dssp= dssp
    
    dssp_feature = np.array(matched_dssp)
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

    return dssp_feature 

def valid_backbone(structure):
    
    
    atom_types = np.array(structure["atom_type"])    
    atom_indices = np.logical_or(atom_types == "CA", atom_types == "N")    
    atom_indices = np.logical_or(atom_types == "C", atom_indices) 
    coordinates = np.array(structure["coordinates"])[atom_indices]
   
    if np.any(atom_types[atom_indices][0::3] != "N"):     
        print(atom_types[atom_indices][0::3])
    
    if np.any(atom_types[atom_indices][1::3] != "CA"):      
        print(atom_types[atom_indices][1::3])    
    if np.any(atom_types[atom_indices][2::3] != "C"):        
        print(atom_types[atom_indices][2::3])        
   
    if len(coordinates) != 3*len(np.unique(structure["residue_number"])):        
        return False
    
    return True

   
def get_nearest_neighbor_distances(protein_structure):
    
    # find the k nearest neighbors for each residue and their distance
    atom_types = np.array(protein_structure["atom_type"])        
    ca_indices = atom_types == "CA"        
    ca_coords = np.array(protein_structure["coordinates"])[ca_indices]    
    pairwise_distances = squareform(pdist(ca_coords))    
    #print(pairwise_distances.shape)
    distances = np.zeros((len(ca_coords), nearest_neighbors))    
    neighbor_indices = np.zeros((len(ca_coords), nearest_neighbors))
    
    #if pairwise_distances.shape[1]=nearest_neighbors:
    
    for index, row in enumerate(pairwise_distances): 
        #print('1',row.shape)        
        if len(row)==nearest_neighbors:
           current_indices = np.argpartition(row, nearest_neighbors-1)        
        else:
           current_indices = np.argpartition(row, nearest_neighbors)
        neighbor_indices[index] = current_indices[:nearest_neighbors]        
        distances[index] = row[current_indices[:nearest_neighbors]]
    #print(distances.shape)   
    return distances, neighbor_indices

def get_rotamer_locations_and_distances(protein_structure):
    
    # get the location and distance of the sidechain centroid from each residue
    
    atom_types = np.array(protein_structure["atom_type"])        
    ca_indices = atom_types == "CA"    
    ca_coords = np.array(protein_structure["coordinates"])[ca_indices]    
    atom_indices = np.logical_and(atom_types != "CA", atom_types != "N")
    atom_indices = np.logical_and(atom_types != "C", atom_indices)    
    atom_indices = np.logical_and(atom_types != "O", atom_indices)    
    atom_coords = np.array(protein_structure["coordinates"])[atom_indices]    
    residue_side_chain_numbers = np.array(protein_structure["residue_number"])[atom_indices]    
    residue_ca_numbers = np.array(protein_structure["residue_number"])[ca_indices]
   
    rotamer_locations = []
    rotamer_distances = []
    
    for number in np.unique(residue_ca_numbers):       
        if len(atom_coords[residue_side_chain_numbers == number]) == 0:           
            rotamer_locations.append(ca_coords[list(residue_ca_numbers).index(number)])           
            rotamer_distances.append(0)
            
            continue
        
        side_chain_centroid = np.mean(atom_coords[residue_side_chain_numbers == number], axis=0)
        
        ca_coord = ca_coords[list(residue_ca_numbers).index(number)]        
        rotamer_locations.append(side_chain_centroid)
        rotamer_distances.append(np.linalg.norm(ca_coord-side_chain_centroid))
            
    return rotamer_locations, rotamer_distances
  

def get_orientation_features(protein_structure, neighbor_indices,
                              rotamer_locations):
    
    # get orientation features for each node/edge corresponding to residues
    # in the protein     
    atom_types = np.array(protein_structure["atom_type"])        
    ca_indices = atom_types == "CA"   
    ca_coords = np.array(protein_structure["coordinates"])[ca_indices]
    
    # get pairwise vectors between adjancent alpha carbons   
    virtual_bonds = ca_coords[1:] - ca_coords[:-1]   
    b1 = virtual_bonds[:-1]    
    b0 = virtual_bonds[1:]
    # get norm of vectors before and after each residue
    n = np.cross(b1, b0)     
    n = n/np.linalg.norm(n, axis=1).reshape(len(n),1)    
    # get negative bisector
    o = b1 - b0   
    o = o/np.linalg.norm(o, axis=1).reshape(len(o),1)    
    O = np.concatenate((o, n, np.cross(o, n)), axis=1)    
    # add 0s for the first and last residue
    O = np.pad(O, ((1, 1),(0,0)), mode="constant", constant_values=0)    
    neighbor_directions = np.zeros(neighbor_indices.shape + (3,))    
    neighbor_orientations = np.zeros(neighbor_indices.shape + (4,))    
    rotamer_directions = np.zeros((len(rotamer_locations), 3))
       
    
    for residue_number, orientation in enumerate(O):
             
        # get neighbor indicies
        
        adjacent_indices = neighbor_indices[residue_number].astype(int)        
        # calculate pairwise CA directios
        displacement = ca_coords[residue_number] - ca_coords[adjacent_indices]        
        directions = np.matmul(orientation.reshape((3,3)), displacement.T).T        
        norm = np.linalg.norm(directions, axis=1).reshape(len(directions),1)        
        directions = np.divide(directions,norm,where=norm!=0)       
        neighbor_directions[residue_number] = directions
        
        # calculate rotamer centroid direction
        displacement = ca_coords[residue_number] - rotamer_locations[residue_number]        
        directions = np.matmul(orientation.reshape((3,3)), displacement.T).T       
        norm = np.linalg.norm(directions*1.0)        
        directions = np.divide(directions,norm,where=norm!=0)       
        rotamer_directions[residue_number] = directions        
        # calculate relative orientation of coordinate systems
        neighbor_matricies = O[adjacent_indices].reshape((-1, 3, 3))              
        rotation_matricies = np.matmul(orientation.reshape((3,3)),np.transpose(neighbor_matricies, (0,2,1)))       
        rotation_matricies = Rotation.from_matrix(rotation_matricies)        
        rotation_matricies = rotation_matricies.as_quat()        
        norm = np.linalg.norm(rotation_matricies, axis=1).reshape(len(rotation_matricies),1)        
        rotation_matricies = np.divide(rotation_matricies,norm,where=norm!=0)        
        neighbor_orientations[residue_number] = rotation_matricies
         
    return neighbor_directions, neighbor_orientations, rotamer_directions


def rbf(distances, rotamer_distance=False, binding_site_distance=False):
    
    # lift the input distances to a radial basis
    
    if not rotamer_distance and not binding_site_distance:
        min_dist = min_dis
        max_dist = max_dis
        counts = num_rbf
    elif rotamer_distance:
        min_dist = min_dis_rot
        max_dist = max_dis_rot
        counts = num_rbf_rot
    elif binding_site_distance:
        min_dist = min_dis_bind
        max_dist = max_dis_bind
        counts = num_rbf_bind
        
    means = np.linspace(min_dist, max_dist, counts)    
    std = (max_dist - min_dist)/counts    
    distances = np.repeat(np.expand_dims(distances, axis=len(distances.shape)), 
                          counts, len(distances.shape))   
    distances = np.exp(-((distances-means)/std)**2)
    
    return distances


def positional_embedding(indices):
    
    # performs a positional embedding for the edges of the input
    
    differences = indices - np.arange(len(indices)).reshape(len(indices),1)   
    result = np.exp(np.arange(0, number_pos_encoding,2)*-1*(np.log(10000)/number_pos_encoding))    
    differences = np.repeat(np.expand_dims(differences, axis=len(differences.shape)),
                            number_pos_encoding/2, len(differences.shape))    
    result = differences*result    
    result = np.concatenate((np.sin(result), np.cos(result)),2)
    
    return result
    
    
    
def get_neighbor_features(nodes, neighbor_indices):
    #extended_node_features =  * 4  # Total number of features after adding statistics
    nodes_with_stats = np.zeros((len(nodes), nodes.shape[1]-20,5))  
    #print(nodes_with_stats.shape) #(256, 23, 4)
    
    for i, node_index in enumerate(neighbor_indices):
            #print(node_index.shape)
            node_values = np.zeros((nearest_neighbors-1, nodes.shape[1]-20))
            #print(f'note index {i}',node_index.shape)
            
            #delete self,
            node_delete=list(set(node_index)-set([i]))
            node_delete = [int(x) for x in node_delete]
            #print(len(node_delete))
            
            # Neighbor features
            for j, neighbor_index in enumerate(node_delete):
                
                node_values[j] = nodes[neighbor_index,20:]
                
            #print(node_values.shape)
            # Calculate statistics
            max_values = np.max(node_values, axis=0)
            max_values = max_values.reshape(-1,1)
            #print(max_values.shape)#(23, 1)
            min_values = np.min(node_values, axis=0)
            min_values = min_values.reshape(-1,1)
            mean_values = np.mean(node_values, axis=0)
            mean_values = mean_values.reshape(-1,1)
            median_values = np.median(node_values, axis=0)
            median_values = median_values.reshape(-1,1)
            
            node_self=(np.transpose(nodes[i,20:])).reshape(-1,1)
            #print(node_self.shape)
            node_stats = np.concatenate((node_self,max_values, min_values, mean_values, median_values),axis=1)
            #print(node_stats.shape)
            nodes_with_stats[i] = node_stats
    
    #print(nodes_with_stats.shape)
    
    return nodes_with_stats


def get_features(pro_seq,pdbid_chain,PDB_file, protein_chain_id):
    
    parser = PDB.PDBParser()

    # read in structure 
    
    structure = parser.get_structure("inputer_protein", PDB_file)

    if not structure.get_list()[0].has_id(protein_chain_id):
        raise ValueError("The chain is not in the provided PDB file")

    protein_chain = structure.get_list()[0].__getitem__(str(protein_chain_id))
    protein_residues = protein_chain.get_list()
    
    prot_inform = {}
    prot_inform["coordinates"] = []
    prot_inform["residue_number"] = []
    prot_inform["residue_type"] = []
    prot_inform["atom_type"] = []
    prot_inform["actual_number"] = []
    ''' coor,20 animo acid, Physicochemical,Interface_propensity'''
    n_features = 20+7+4
    
    nodes_phe_chi_IP=[]
    pdb_seq=[]
    for index, res in enumerate(protein_residues):
        
        ''' skip non standard residues'''
        if not PDB.Polypeptide.is_aa(res.get_resname(), standard=True):           
            
           continue   
                   
        resname = res.get_resname()
        #print(resname)
        resnumber = index
        all_atoms = [atom.get_name().strip() for atom in res.get_atoms()]        
        
        # skip residues lacking a backbone antom
        if "C" not in all_atoms or "CA" not in all_atoms or "N" not in all_atoms:
            continue
          
        n_atom = []
        ca_atom = []
        c_atom = []
        for a_i, atom in enumerate(res.get_atoms()):
            if atom.get_name().strip() != "H":
                
                if atom.get_name().strip() == "N":
                    n_atom.append(atom.get_coord())
                    n_atom.append(resnumber)
                    n_atom.append(res.get_id()[1])
                    n_atom.append(resname)
                    n_atom.append(atom.get_name().strip())
                    continue
                elif atom.get_name().strip() == "CA":
                    ca_atom.append(atom.get_coord())
                    ca_atom.append(resnumber)
                    ca_atom.append(res.get_id()[1])
                    ca_atom.append(resname)
                    ca_atom.append(atom.get_name().strip())
                    
                    features = [0]*n_features
                    temp_target = np.zeros(20)
                    residue=three_to_one(resname)
                    #print(residue)
                    ''' no standard residue is denoted as G (H)'''
                    if residue == "X":
                       resname='GLY'     
                    temp_target[PDB.Polypeptide.d3_to_index[resname]] = 1    
                    features[0:20] = temp_target                    
                    
                    phe_chi=AA_dict[residue]
                    features[20:27] = phe_chi
                    
                    P_residue=IP_dict[residue]
                    #P_feature=[sum(P_residue)/len(P_residue),max(P_residue),min(P_residue)]
                    features[27:31] = P_residue
                    
                    pdb_seq.append(residue)
                                           
                    nodes_phe_chi_IP.append(features)     
                                   
                    continue
                    
                elif atom.get_name().strip() == "C":
                    c_atom.append(atom.get_coord())
                    c_atom.append(resnumber)
                    c_atom.append(res.get_id()[1])
                    c_atom.append(resname)
                    c_atom.append(atom.get_name().strip())
                    continue
                prot_inform["coordinates"].append(atom.get_coord())
                prot_inform["residue_number"].append(resnumber)
                prot_inform["actual_number"].append(res.get_id()[1])
                prot_inform["residue_type"].append(resname)
                prot_inform["atom_type"].append(atom.get_name().strip())
                
                
        prot_inform["coordinates"].append(n_atom[0])
        prot_inform["residue_number"].append(n_atom[1])
        prot_inform["actual_number"].append(n_atom[2])
        prot_inform["residue_type"].append(n_atom[3])
        prot_inform["atom_type"].append(n_atom[4])
        
        prot_inform["coordinates"].append(ca_atom[0])
        prot_inform["residue_number"].append(ca_atom[1])
        prot_inform["actual_number"].append(ca_atom[2])
        prot_inform["residue_type"].append(ca_atom[3])
        prot_inform["atom_type"].append(ca_atom[4])
        
        prot_inform["coordinates"].append(c_atom[0])
        prot_inform["residue_number"].append(c_atom[1])
        prot_inform["actual_number"].append(c_atom[2])
        prot_inform["residue_type"].append(c_atom[3])
        prot_inform["atom_type"].append(c_atom[4])
                

                       
    # check that the backbones are valid for both molecules
    
    if not valid_backbone(prot_inform):
        raise ValueError("The backbone of the specified chain is not well formed")
        return

    # define array for node and edge features
    node = np.zeros((len(np.unique(prot_inform["residue_number"])), node_features))
    

    edges = np.zeros((len(np.unique(prot_inform["residue_number"])), nearest_neighbors,edge_features))

    # encode sequence of amino acids as node feature   
    amino_acid_index = np.array(prot_inform["atom_type"]) == "CA"   
    amino_acid_sequence = np.array(prot_inform["residue_type"])[amino_acid_index]
            
 
    node[:, 0:n_features] = nodes_phe_chi_IP
    pdb_seq =''.join(aa for aa in pdb_seq)
    #print(0)
    #dssp
    dsspseq=(dssp_dict[pdbid_chain])[0]
    dssp=(dssp_dict[pdbid_chain])[1]       
    dssp_feature=match_dssp(dsspseq, dssp, pdb_seq)
    
    if len(dssp_feature)!=len(pdb_seq):
       print(f'{pdbfile} dssp not align!')
    else:
       ''' 
       20 animo acid, Physicochemical,Interface_propensity,dssp
       20+7+4+13=44          
       '''
       nodes=np.hstack([node, dssp_feature])       
   
     # get distances and coordinates for orientation features
    neighbor_distances, neighbor_indices= get_nearest_neighbor_distances( prot_inform)

    adjusted_neighbor_distances = rbf(neighbor_distances)
    #num_rbf=16
    edges[:, :, 0:num_rbf] = adjusted_neighbor_distances

    # get rotamer distances and coordinates for orientation features
    rotamer_locations, rotamer_distances = get_rotamer_locations_and_distances(prot_inform)    
    # get orientation features
    neighbor_directions, neighbor_orientations, rotamer_directions = get_orientation_features(
            prot_inform, neighbor_indices, rotamer_locations)
    
    edges[:, :, num_rbf:num_rbf+3] = neighbor_directions
    edges[:, :, num_rbf+3:num_rbf+7] = neighbor_orientations

    
    edge_embeddings = positional_embedding(neighbor_indices)
    edges[:, :, num_rbf+7: num_rbf+7+number_pos_encoding] = edge_embeddings

    
    
    nodes_with_stats = get_neighbor_features(nodes, neighbor_indices)


    ''' label '''
    #print(len(pro_seq))
    #print(len(pdb_seq))
    
    aligs = pairwise2.align.globalxx(pdb_seq,pro_seq)
    pdbseq = aligs[0].seqA
    proseq = aligs[0].seqB
    label_pdb=np.zeros(len(pdbseq))
    original_label=ori_label
    #print(len(pdbseq))
    #print(len(proseq))
    #print(len(ori_label))    
    for i in range(len(pdbseq)):
       if pdbseq[i] == "-" :
          label_pdb[i]=2 
       else:
          if pdbseq[i] != "-" and proseq[i] != "-": 
             label_pdb[i]=original_label[i]
          else:
          	 label_pdb[i]=0
          	 original_label = original_label[:i] + '0' + original_label[i:]       
    filtered_list = [value for value in label_pdb if value != 2]
    label=np.array(filtered_list)
    
    pdbseq_out.write(f'{pdbid_chain}#{pdb_seq}\n')     
    #print(label)
    #print((label != np.zeros(len(pdbseq))).any())
    if len(label) > 0 and (label != np.zeros(len(pdb_seq))).any():
       savefile=[pdb_seq, label, nodes_with_stats, edges, neighbor_indices]
    else:
       savefile=[]    
    return  savefile


PDB_path= sys.argv[1]
Type=sys.argv[2]   
dataset='./example/'+sys.argv[3]
nearest_neighbors = int(sys.argv[4])

#testdata=sys.argv[5]
#traindata=sys.argv[6]

outpath=f'./{Type}_feature_results/'
os.system(f'mkdir {outpath}')
    
with open(f"./example/features/pro_dssp_dict{Type}", "rb") as f:
    dssp_dict = pickle.load(f)
with open("./codes/AA_dict", "rb") as f:
    AA_dict = pickle.load(f)
with open(f"./codes/protein_protein_dict", "rb") as f:
    IP_dict = pickle.load(f)     

f.close()  


num_rbf = 16
num_rbf_bind = 16
num_rbf_rot = 3        
number_pos_encoding = 16
edge_features = 3 + 4 +num_rbf + number_pos_encoding        

min_dis = 0
max_dis = 20        
min_dis_bind = 0
max_dis_bind = 100        
min_dis_rot = 0
max_dis_rot = 6
        
node_features = 20+7+ 4 



#
#pdbseq_out=open(f'pdbseq_{Type}.txt','a')
protein_dict={}

for i  in open(dataset):

    temp=i.strip().split('#')
    fileid=temp[0]    
    pro=fileid[4:5]        
    ori_label=temp[2].strip()
    pro_seq=temp[1].strip()    
    #label_index= [i for i, value in enumerate(ori_label) if value == '1']

    name=fileid[0:4]+'_'+ fileid[4]      
    pdbfile = f'{PDB_path}/{name}.pdb'
    print(f' {temp[0]}  computed')
    
    pro_inform =get_features(pro_seq,temp[0],pdbfile, str(pro))
      
   
    if len(pro_inform) > 0: 
       protein_dict[temp[0]]= pro_inform
       print(f' {temp[0]}  succeful')
    else:
       break 
'''
   key:4i2wA
   value=[pdb_seq, label, nodes_with_stats, edges, neighbor_indices] 
    
'''  
print(len(protein_dict))


with open(f'{outpath}protein_{Type}_{str(nearest_neighbors)}_dict', "wb") as f:
    pickle.dump(protein_dict,f)

#with open(f'{outpath}protein_{Type}_{str(nearest_neighbors)}_dict', "rb") as f:
#    protein_dict=pickle.load(f)

#split_data(protein_dict)	
os.system('mv {outpath} example/features')
#pdbseq_out.close()










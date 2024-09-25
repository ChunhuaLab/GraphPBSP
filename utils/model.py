import torch
import  torch.nn  as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, NNConv, FeaStConv
#from mgat import MGAT
import numpy as np
from torch_geometric.nn import global_mean_pool

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class FNN(nn.Module):
    
    def __init__(self, d_in, d_hid, dropout=0.2):
        super().__init__()
        
        self.layer_1 = nn.Conv1d(d_in, d_hid,1)
        self.layer_2 = nn.Conv1d(d_hid, d_in,1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_in)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        residual = x 
        output = self.layer_1(x.transpose(1, 2))        
        output = self.relu(output)        
        output = self.layer_2(output)        
        output = self.dropout(output)        
        output = self.layer_norm(output.transpose(1, 2)+residual)
        
        return output

class MLP(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super(MLP, self).__init__()
        size=int(input_dim/2)
        self.fc1 = nn.Linear(input_dim, size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(size, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        #print(x.shape)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MGAT(nn.Module):
    
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.2):
        super().__init__()
              
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
              
        self.W_Q = nn.Linear(d_model, n_head*d_k)        
        self.W_K = nn.Linear(d_model*2, n_head*d_k)
        self.W_V = nn.Linear(d_model*2, n_head*d_v)
        self.W_O = nn.Linear(n_head*d_v, d_model)
               
        self.softmax = nn.Softmax(dim=-1)        
        self.layer_norm = nn.LayerNorm(d_model)       
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, nodes, edges):
        
        
        n_batch, n_nodes, n_neighbors = edges.shape[:3]        
        Q = self.W_Q(nodes).view([n_batch, n_nodes, 1, self.n_head, 1, self.d_k])
        K = self.W_K(edges).view([n_batch, n_nodes, n_neighbors, self.n_head, self.d_k, 1])
        
        attention = torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, self.n_head]).transpose(-2,-1)       
        attention = attention /np.sqrt(self.d_k)        
        attention = self.softmax(attention)
            
        V = self.W_V(edges).view([n_batch, n_nodes, n_neighbors, self.n_head, self.d_v]).transpose(2,3)        
        attention = attention.unsqueeze(-2)               
            
        output = torch.matmul(attention, V).view([n_batch, n_nodes, self.d_v*self.n_head])        
        output = self.W_O(output)       
        output = self.dropout(output)        
        output = self.layer_norm(output + nodes)
                          
        #attention = attention.squeeze(-2).transpose(-2,-1)
        
        #return output, attention
        return output



class PredModel(torch.nn.Module):
    def __init__(self, nheads,d_model,nf,n_out,node_features,edge_features,dropout=0.1):
        super(PredModel, self).__init__()

        self.edge_embedding = nn.Linear(edge_features, d_model)
        self.node_embedding = nn.Linear(node_features*2, d_model)
        self.node = nn.Linear(node_features*5, node_features) 
        self.protT5_embedding = nn.Linear(1024, node_features)
        self.Dropout1 = nn.Dropout(dropout)
        self.Dropout2 = nn.Dropout(dropout)                      
        self.FNN = FNN(d_model, d_model)
        
        GAT = GATConv
        self.MGAT = MGAT(nheads,d_model, nf, n_out,dropout=dropout)
        self.line = torch.nn.Linear(node_features, nf)
        self.GAT1 = GAT(nf, int(nf/nheads), heads=nheads)        
        self.lin1 = torch.nn.Linear(nf, nf)
        
        #self.conv3 = GAT(nf, int(nf/nheads),  heads=nheads)
        #self.lin3 = torch.nn.Linear(nf, nf)        
        self.GAT2 = GAT(nf, int(n_out/nheads),  heads=nheads)        
        self.lin2 = torch.nn.Linear(nf, n_out)
                   
        self.mlp =MLP(n_out,2)
        self.layer_norm = nn.LayerNorm(nf)             
        self.softmax_nodes =nn.LogSoftmax(dim=-1)
        

    
    def forward(self, protT5,X, E, Neighb): 
        #([904, 1024]) torch.Size([904, 23, 5]) torch.Size([904, 30, 39]) torch.Size([904, 30])
        #print( protT5.shape,X.shape, E.shape, Neighb.shape)
        #torch.Size([1, 904, 1024]) torch.Size([1, 904, 23, 5]) torch.Size([1, 904, 30, 39]) torch.Size([1, 904, 30])
        
        nodes= X.view(X.shape[0],X.shape[1], -1)        
        #print(nodes.shape)#torch.Size([1, 904, 115])
        nodes = self.node(nodes)
        #print(nodes.shape)#torch.Size([1, 904, 23])
        edge_embedding = self.edge_embedding(E)
        #print(edge_embedding.shape)#torch.Size([1, 904, 30, 16])
        protT5_embedding=self.protT5_embedding(protT5)
        #print(protT5_embedding.shape)#torch.Size([1, 904, 23])
       
        nodes_embedding = self.node_embedding(torch.cat([nodes, protT5_embedding], dim=-1))
        
        neighbors= Neighb.view((Neighb.shape[0], -1))
        neighbors= neighbors.unsqueeze(-1).expand(-1, -1, nodes_embedding.size(2)).to(torch.int64)
        neighbor_features = torch.gather(nodes_embedding, 1, neighbors)
        neighbor_features = neighbor_features.view(list(Neighb.shape)[:3] + [-1])        
        
        edge_neighbor=torch.cat([edge_embedding, neighbor_features], -1)
       
        node_enc = self.Dropout1(nodes_embedding)        
        edge_input = self.Dropout2(edge_neighbor)        
        
        node_enc = self.MGAT(node_enc, edge_input)
        
        node_enc = self.FNN(node_enc)
        

        #x = F.elu(self.GAT2(x, edge_index) + self.lin2(x))
#        x = F.dropout(x, p=0.1, training=self.training)
#        x1 = F.elu(self.conv3(x, edge_index) + self.lin3(x))        
#        x1 = F.dropout(x1, p=0.1, training=self.training)
#        x2 = F.elu(self.conv4(x1, edge_index) + self.lin4(x1)) # K, d
#        x3 = self.layer_norm(x2)

     
        return node_enc



    def get_pred(self,protT5,X, E, Neighb):
        with torch.no_grad():
            output = self.forward(protT5,X, E, Neighb)
        pred = self.mlp(output)
        return pred    
    




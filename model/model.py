import torch
from torch import Tensor
from torch_geometric.nn import (SAGEConv,
                                to_hetero, 
                                GENConv,
                                TransformerConv)
import torch.nn.functional as F
from torch_geometric.nn import StdAggregation
from torch_geometric.data import HeteroData
import copy
import torch.nn as nn

class GNN(torch.nn.Module):

    def __init__(self, hidden_channels, aggr, aggr_kwargs=None):
        super().__init__()
        num_aggr = len(aggr)
        self.conv1 = GENConv((-1, -1), 
                             hidden_channels, 
                             aggr=aggr, 
                             aggr_kwargs=aggr_kwargs)

        self.conv2 = SAGEConv(hidden_channels * num_aggr, 
                              hidden_channels, 
                              aggr=copy.deepcopy(aggr), 
                              aggr_kwargs=aggr_kwargs)
        
        self.conv3 = TransformerConv(hidden_channels * num_aggr, 
                                     hidden_channels, 
                                     aggr=copy.deepcopy(aggr), 
                                     aggr_kwargs=aggr_kwargs)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


class Classifier(torch.nn.Module):

    def __init__(self, hidden_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(2 * hidden_channels, 
                                   hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 
                                   1)

    def forward(self, 
                x_ncRNA: Tensor, 
                x_Protein: Tensor, 
                edge_label_index: Tensor) -> Tensor:
        
        edge_feat_ncRNA = x_ncRNA[edge_label_index[0]]
        edge_feat_Protein = x_Protein[edge_label_index[1]]

        edge_feat = torch.cat([edge_feat_ncRNA, edge_feat_Protein], dim=-1)

        x = self.fc1(edge_feat)
        x = F.relu(x)
        x = self.fc2(x).squeeze()
        return x

class Model(torch.nn.Module):

    def __init__(self, hidden_channels, aggr, metadata, num_nodes_dict):
        super().__init__()
        self.hidden_channels = hidden_channels
            
        # Linear transformations for lncRNA and Protein kmers
        self.ncRNA_lin_5mer = torch.nn.Linear(1024, hidden_channels)
        self.ncRNA_lin_4mer = torch.nn.Linear(256, hidden_channels)
        # self.ncRNA_lin_3mer = torch.nn.Linear(64, hidden_channels)

        self.Protein_lin_5mer = torch.nn.Linear(16807, hidden_channels)
        # self.Protein_lin_4mer = torch.nn.Linear(2401, hidden_channels)
        # self.Protein_lin_3mer = torch.nn.Linear(343, hidden_channels)
        # self.Protein_lin_2mer = torch.nn.Linear(49, hidden_channels)

        # Motifs and Distance Map features
        self.ncRNA_lin_motif = torch.nn.Linear(1194,hidden_channels)
        self.Protein_DistanceMap = torch.nn.Linear(64, hidden_channels)

        # Convolutional layers for Protein 3D Structure
        self.protein_conv1 = nn.Conv1d(in_channels=3, 
                                        out_channels=32, 
                                        kernel_size=3, 
                                        stride=1, 
                                        padding=1)
        self.protein_pool = nn.MaxPool1d(kernel_size=2, 
                                        stride=2, 
                                        padding=0)
        self.protein_conv2 = nn.Conv1d(in_channels=32, 
                                        out_channels=64, 
                                        kernel_size=3, 
                                        stride=1, 
                                        padding=1)
        self.fc1 = nn.Linear(16000, hidden_channels)
            
        # Embeddings for nodes
        self.ncRNA_emb = torch.nn.Embedding(num_nodes_dict['ncRNA'], hidden_channels)
        self.Protein_emb = torch.nn.Embedding(num_nodes_dict['Protein'], hidden_channels)
            
        # GNN Layers
        self.gnn = GNN(hidden_channels, aggr, aggr_kwargs=None)
        self.gnn = to_hetero(self.gnn, metadata=metadata)
            
        # Classifier
        self.classifier = Classifier(hidden_channels)


    def forward(self, data: HeteroData) -> Tensor:

        protein_5mer = data['Protein'].x[:,:16807]

        feat_contact = data['Protein'].x[:,16807:16871]

        # Process Protein 3D Structure through CNN layers
        protein3D_flat = data['Protein'].x[:,16871:]
        protein3D = protein3D_flat.reshape(-1, 3, 1000)
        protein_x = self.protein_pool(F.relu(self.protein_conv1(protein3D)))
        protein_x = self.protein_pool(F.relu(self.protein_conv2(protein_x)))
        protein_x = protein_x.view( protein_x.size(0),-1)  # Flatten
        protein3D_Structure = self.fc1(protein_x)        

        rna_4mer = data['ncRNA'].x[:,:256]
        rna_5mer = data['ncRNA'].x[:,256:1280]

        rna_motif = data['ncRNA'].x[:,1280:]
                
        # Prepare node feature dictionaries with concatenated linear and embedding features
        x_dict = {
            'ncRNA': torch.cat((    
                                # F.normalize(self.ncRNA_lin_3mer(rna_3mer),dim=1),                          
                                F.normalize(self.ncRNA_lin_4mer(rna_4mer),dim=1),
                                F.normalize(self.ncRNA_lin_5mer(rna_5mer ),dim=1),
                                F.normalize(self.ncRNA_lin_motif(rna_motif)),
                                F.normalize(self.ncRNA_emb(data['ncRNA'].node_id),dim=1)
                                )),
            'Protein': torch.cat((
                                    # F.normalize(self.Protein_lin_2mer(protein_2mer),dim=1) ,
                                    # F.normalize(self.Protein_lin_3mer(protein_3mer),dim=1) ,
                                    # F.normalize(self.Protein_lin_4mer(protein_4mer),dim=1) ,
                                    F.normalize(self.Protein_lin_5mer(protein_5mer),dim=1) ,
                                    F.normalize(protein3D_Structure,dim=1),
                                    F.normalize(self.Protein_DistanceMap(feat_contact),dim=1) ,
                                    F.normalize(self.Protein_emb(data['Protein'].node_id),dim=1),
                                                                                            )),
        } 
            
        # GNN Forward Pass
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        # Classification
        pred = self.classifier(
                x_dict['ncRNA'],
                x_dict['Protein'],
                data['ncRNA', 'to', 'Protein'].edge_label_index
        )
        
        return pred

# Define aggregation method
aggr = [StdAggregation()]  

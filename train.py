import json
import pandas as pd
import numpy as np
import torch
from matplotlib.colors import LogNorm
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.nn import (SAGEConv,
                                to_hetero, 
                                GENConv,
                                TransformerConv)
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import torch.nn as nn
import random
from torch_geometric.nn import StdAggregation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix,
                             roc_auc_score, 
                             accuracy_score,
                             recall_score, 
                             precision_score,
                             matthews_corrcoef,
                             ConfusionMatrixDisplay)
import argparse
import sys
import tqdm
from torch.optim.lr_scheduler import StepLR
from statistics import mean 
import copy
from datetime import datetime
import os

from utils.Earlystopper import EarlyStopping
from utils.functions import datapreparation
from model.model import (Classifier,
                         GNN,
                         Model)
from utils.figures import (ACC_per_RP,
                           ACC_per_N,
                           tensor_to_list, 
                           ACC_per_P)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NPInter5',
                        help='Choose among NPInter5, NPInter2, RPI369, and RPI7317.')
    parser.add_argument('--Neg_method', type=str, default='BalancedNegGen', 
                        help='Choose negative generation method among BalancedNegGen, Random, and Distance (only for NPInter5).')

    parser.add_argument('--Predicting', type=bool, default=False, 
                        help='This function, after training model predicts new interactions and saves them in a json file')

    parser.add_argument('--proteinkmer', type=bool, default=True,
                        help='If you want to use kmer for lncRNA.')
    parser.add_argument('--lncRNAkmer', type=bool, default=True,
                        help='If you want to use kmer for proteins.')
    parser.add_argument('--motif', type=bool, default=True,
                        help='If you want to use lncRNA motifs.')
    parser.add_argument('--pro3D', type=bool, default=True,
                        help='If you want to use protein 3D structures (This is only availabe for NPInter5).')
    parser.add_argument('--distance', type=bool, default=True,
                        help='If you want to use distance map for proteins (This is only availabe for NPInter5).')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='gamma')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Dimension of representations')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for GNN')
    parser.add_argument('--seeds', type=list, default=[50,100,150,200,250],
                        help='Seeds for cross validation. Length of seed list would be number of cross validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of test data')
    parser.add_argument('--val_ratio', type = float, default= 0.1,
                        help='Ratio of validation data')
    parser.add_argument('--num_neighbors', type = list, default= [20,10],
                        help='num of neighbors to load batches of the graph')

    # Opening file for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join("results", timestamp)
    os.makedirs(save_folder, exist_ok=True)

    if 'ipykernel' in sys.argv[0]:
        # Avoid parsing unwanted arguments in Jupyter
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # Opening a txt file to write model results and configurations
    txt_path = os.path.join(save_folder, "model_info_result.txt")
    def txt_writer(text):
        with open(txt_path , "a") as f:
            f.write(text)

    # Writing configurations
    txt_writer(f"Dataset: {args.dataset}\n")
    txt_writer(f"Negative Generation Method: {args.Neg_method}\n")
    txt_writer("Utilized Protein Features: ")
    if args.proteinkmer:
        txt_writer("k-mer, ")
    if args.pro3D and args.dataset == 'NPInter5':
        txt_writer("3D Structure, ")
    if args.distance and args.dataset == 'NPInter5':
        txt_writer("Distance Map ")
    txt_writer("\nUtilized lncRNA Features: ")
    if args.lncRNAkmer:
        txt_writer("k-mer, ")
    if args.motif:
        txt_writer("Binding Motifs ")

    ## Reading Features of lncRNAs and Proteins
    # Reading Dataset
    dataset_path = 'data/dataset/'+ args.dataset + '_' + args.Neg_method + '.csv'
    dataset=pd.read_csv(dataset_path)
    # Reading lncRNA k-mer
    rna_kmer_dict = {}
    for k in [4,5]:
        kmer_path = 'data/lncRNA/k-mer/' + args.dataset + '/' + args.dataset + '_lncRNA_' + f'{k}'+ '_mer.json'
        with open(kmer_path) as json_file:
            rna_kmer_dict [f'{k}mer'] = json.load(json_file)
    # Reading protein k-mer
    pro_kmer_dict = {}
    for k in [5]:
        kmer_path = 'data/protein/k-mer/' + args.dataset + '/' + args.dataset + '_protein_' + f'{k}' + '_mer.json'
        with open(kmer_path) as json_file:
            pro_kmer_dict [f'{k}mer'] = json.load(json_file)
    # Reading lncRNA motif
    if args.motif:
        motif_path = 'data/lncRNA/motif/' + args.dataset +'/' + args.dataset +'_lncRNA_motif.json'
        with open(motif_path) as json_file:
            rna_motif_dict = json.load(json_file)
    else:
        rna_motif_dict = {}
    # Reading protein distance map
    if args.distance and args.dataset == 'NPInter5':
        distance_map_path = 'data/protein/distance map/' + args.dataset +'/' + args.dataset +'_distance_map_dic_auto.json'
        with open(distance_map_path) as json_file:
            distance_map_dict_auto = json.load(json_file)
    else:
        distance_map_dict_auto = {}
    # Reading protein 3D structure
    if args.pro3D and args.dataset == 'NPInter5':

        pro3D_path = 'data/protein/3D-structure/' + args.dataset +'/' + args.dataset +'_pro_3D_dic.json'
        with open(pro3D_path) as json_file:
            pro3D_dict = json.load(json_file)
    else:
        pro3D_dict = {}

    # Data Preparation
    print("Loading Data, Be Patient ...")
    ncfeat, tarfeat, edge_index, edge_label, unique_ncID, unique_tarID = datapreparation(dataset = dataset,
                    rna_kmer_dict = rna_kmer_dict,
                    protein_kmer_dict = pro_kmer_dict,
                    rna_motif_dict = rna_motif_dict,
                    distance_map_dict = distance_map_dict_auto,
                    protein_3D_dict = pro3D_dict,
                    args = args)
    print("Loading data is finished.")

    #Bipartite Graph Generation
    data = HeteroData()
    data['ncRNA'].node_id = torch.arange(len(unique_ncID))
    data['Protein'].node_id = torch.arange(len(unique_tarID))
    data['ncRNA'].x=ncfeat
    data['Protein'].x=tarfeat
    data['ncRNA','Protein'].edge_index = edge_index
    data['ncRNA','Protein'].edge_label = edge_label
    data = T.ToUndirected()(data)

    # Device Agnostic Code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    # Define aggregation method
    aggr = [StdAggregation()]  

    seeds = args.seeds
    fold_counter = 0
    MCCs = []
    Precisions = []
    Specificities = []
    Sensitivities = []
    Accuracies = []
    AUCs = []

    txt_writer(f"\nSeeds: {seeds}\n")

    for seed in seeds:
        # Using Seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

        fold_counter += 1
        print("\nTraining, fold number ", fold_counter)
        txt_writer(f"\nTraining, fold number {fold_counter}\n")

        # Spiliting Data into train, test and validation
        transform = T.RandomLinkSplit(
            num_val = args.val_ratio,
            num_test = args.test_ratio,
            is_undirected = True,
            key= 'edge_label',
            add_negative_train_samples=False,
            neg_sampling_ratio = 0,
            edge_types=('ncRNA','to','Protein'),
            rev_edge_types=('Protein','rev_to','ncRNA')
            )
        train_data, val_data, test_data = transform(data)

        # Loading Graph specifically for link prediction
        edge_label_index = train_data['ncRNA','to','Protein'].edge_label_index
        edge_label = train_data['ncRNA','to','Protein'].edge_label
        # train loader
        train_loader = LinkNeighborLoader(
            data = train_data,
            num_neighbors=args.num_neighbors,
            edge_label_index=(('ncRNA','to','Protein'),edge_label_index),
            edge_label=edge_label,
            batch_size = args.batch_size,
            num_workers=8,
            shuffle = True
        )
        # test loader
        edge_label_index = val_data['ncRNA', 'to', 'Protein'].edge_label_index
        edge_label = val_data['ncRNA', 'to', 'Protein'].edge_label
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=args.num_neighbors,
            edge_label_index=(('ncRNA', 'to', 'Protein'), edge_label_index),
            edge_label=edge_label,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=False,
        )

        # Retrieve metadata and node counts
        metadata = train_data.metadata()  # e.g., (['ncRNA', 'Protein'], [('ncRNA', 'to', 'Protein')])
        num_nodes_dict = train_data.num_nodes_dict

        # Initialize the model with metadata and node counts and sending model to device
        model = Model(hidden_channels=args.hidden, aggr=aggr, metadata=metadata, num_nodes_dict=num_nodes_dict)
        model = model.to(device)

        #Defining Optimizer and scheduler for reducing learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) 

        es = EarlyStopping()

        num_epochs = args.epochs
        for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
            total_loss = total_examples = 0
            model.train()

            for sampled_data in train_loader:
                optimizer.zero_grad()
                sampled_data.to(device)
                pred = model(sampled_data).to(float)
                ground_truth = sampled_data['ncRNA', 'to', 'Protein'].edge_label.to(float)
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()

            scheduler.step()
            pred = pred.detach().cpu().numpy()
            ground_truth = ground_truth.detach().cpu().numpy()
            pred_sigmoid_np = torch.sigmoid(torch.from_numpy(pred)).numpy()
            train_auc = roc_auc_score(ground_truth, pred_sigmoid_np)
            train_acc=accuracy_score(ground_truth,np.round(pred_sigmoid_np))
            train_loss = total_loss / total_examples

            #Evaluation
            total_loss = total_examples = 0
            model.eval()
            preds = []
            ground_truths = []
            for sampled_data in val_loader:
                with torch.no_grad():
                    sampled_data.to(device)
                    valpred = model(sampled_data).to(float)
                    valground_truth = sampled_data['ncRNA', 'to', 'Protein'].edge_label.to(float)
                    preds.append(valpred)
                    ground_truths.append(valground_truth)
                    loss = F.binary_cross_entropy_with_logits(valpred, valground_truth)
                    total_loss += float(loss) * valpred.numel()
                    total_examples += valpred.numel()

            val_pred = torch.cat(preds, dim=0).cpu().numpy()
            val_ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
            val_pred_sigmoid_np = torch.sigmoid(torch.from_numpy(val_pred)).numpy()
            val_auc = roc_auc_score(val_ground_truth, val_pred)
            val_acc=accuracy_score(val_ground_truth,np.round(val_pred_sigmoid_np))
            val_loss = total_loss / total_examples
            # Using Early stop function
            if es(model, val_loss):
                print("Early stopping function was activated for this fold")
                break
            
        preds = []
        ground_truths = []
        model.eval()
        for sampled_data in train_loader:
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(model(sampled_data))
                ground_truths.append(sampled_data['ncRNA', 'to', 'Protein'].edge_label)

        train_pred = torch.cat(preds, dim=0).cpu().numpy()
        train_ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        train_pred_sigmoid = torch.sigmoid(torch.from_numpy(train_pred)).numpy()
        auc = roc_auc_score(train_ground_truth, train_pred_sigmoid)
        acc = accuracy_score(train_ground_truth, np.round(train_pred_sigmoid))

        edge_label_index = test_data['ncRNA', 'to', 'Protein'].edge_label_index
        edge_label = test_data['ncRNA', 'to', 'Protein'].edge_label
        test_loader = LinkNeighborLoader(
            data=test_data,
            num_neighbors=args.num_neighbors,
            edge_label_index=(('ncRNA', 'to', 'Protein'), edge_label_index),
            edge_label=edge_label,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
        )

        preds = []
        ground_truths = []
        for sampled_data in test_loader:
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(model(sampled_data))
                ground_truths.append(sampled_data['ncRNA', 'to', 'Protein'].edge_label)

        test_pred_tensor = torch.cat(preds, dim=0).cpu()  
        test_gt_tensor   = torch.cat(ground_truths, dim=0).cpu() 
        test_pred = test_pred_tensor.numpy()
        test_ground_truth = test_gt_tensor.numpy()
        test_pred_sigmoid = torch.sigmoid(test_pred_tensor).numpy()
        auc = roc_auc_score(test_ground_truth, test_pred_sigmoid)
        binary_preds = np.round(test_pred_sigmoid)
        acc = accuracy_score(test_ground_truth, binary_preds)
        sensitivity = recall_score(test_ground_truth, binary_preds)
        precision = precision_score(test_ground_truth, binary_preds)
        mcc = matthews_corrcoef(test_ground_truth, binary_preds)
        tn, fp, fn, tp = confusion_matrix(test_ground_truth, binary_preds).ravel()
        specificity = tn / (tn + fp)

        # Writing model results for each fold
        txt_writer(f"\nResult Metrics for fold number:{fold_counter} \n")
        print(f'AUC: {auc:4f}')
        txt_writer(f'AUC: {auc:4f}\n')
        AUCs.append(auc)
        print(f'Accuracy: {acc:4f}')
        txt_writer(f'Accuracy: {acc:4f}\n')
        Accuracies.append(acc)
        print(f'Sensitivity (Recall): {sensitivity:4f}')
        txt_writer(f'Sensitivity (Recall): {sensitivity:4f}\n')
        Sensitivities.append(sensitivity)
        print(f'Specificity: {specificity:4f}')
        txt_writer(f'Specificity: {specificity:4f}\n')
        Specificities.append(specificity)
        print(f'Precision: {precision:4f}')
        txt_writer(f'Precision: {precision:4f}\n')
        Precisions.append(precision)
        print(f'Matthews Correlation Coefficient (MCC): {mcc:4f}')
        txt_writer(f'Matthews Correlation Coefficient (MCC): {mcc:4f}\n')
        MCCs.append(mcc)
    
    # Writing cross validation results
    print('\nCross Validation Metrics:')
    txt_writer('\nCross Validation Metrics:\n')

    AUC_mean = mean(AUCs)
    print(f'AUC: {AUC_mean:4f}')
    txt_writer(f'AUC: {AUC_mean}\n')
    Accuracies_mean = mean(Accuracies)
    print(f'Accuracy: {Accuracies_mean:4f}')
    txt_writer(f'Accuracy: {Accuracies_mean}\n')
    Sensitivities_mean = mean(Sensitivities)
    print(f'Sensitivity (Recall): {Sensitivities_mean:4f}')
    txt_writer(f'Sensitivity (Recall): {Sensitivities_mean}\n')
    Specificities_mean = mean(Specificities)
    print(f'Specificity: {Specificities_mean:4f}')
    txt_writer(f'Specificity: {Specificities_mean}\n')
    Precisions_mean = mean(Precisions)
    print(f'Precision: {Precisions_mean:4f}')
    txt_writer(f'Precision: {Precisions_mean}\n')
    MCCs_mean = mean(MCCs)
    print(f'Matthews Correlation Coefficient (MCC): {MCCs_mean:4f}')
    txt_writer(f'Matthews Correlation Coefficient (MCC): {MCCs_mean}\n')
    print('Training is finished')
    print('Saving the results ...')

    #Saving the model and write its configuration
    save_folder2 = os.path.join("saved model", args.dataset , args.Neg_method)
    os.makedirs(save_folder2, exist_ok=True)
    model_path = os.path.join(save_folder2, "model_weights.pth")
    torch.save(model.state_dict(), model_path)
    txt_path2 = os.path.join(save_folder2, "model_configuration.txt")

    def txt_writer2(text):
        with open(txt_path2 , "a") as f:
            f.write(text)

    txt_writer2(f"Dataset: {args.dataset}\n")
    txt_writer2(f"Negative Generation Method: {args.Neg_method}\n")
    txt_writer2(f"Batch Size: {args.batch_size}\n")
    txt_writer2(f"Number of Hidden Channels: {args.hidden}\n")
    txt_writer2(f"Number of Neighbors: {args.num_neighbors}\n")
    txt_writer2("Utilized Protein Features: ")

    if args.proteinkmer:
        txt_writer2("k-mer, ")
    if args.pro3D and args.dataset == 'NPInter5':
        txt_writer2("3D Structure, ")
    if args.distance and args.dataset == 'NPInter5':
        txt_writer2("Distance Map ")
    txt_writer2("\nUtilized lncRNA Features: ")
    if args.lncRNAkmer:
        txt_writer2("k-mer, ")
    if args.motif:
        txt_writer2("Binding Motifs ")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    # Saving Confusion Matrix
    cm = confusion_matrix(
                        test_ground_truth,
                        torch.round(torch.sigmoid(torch.tensor(test_pred))))
    cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=[0, 1])
    disp.plot(values_format='.2f', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Row Percentage)", fontname='Times New Roman', color='black')
    save_path = os.path.join(save_folder, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

    train_edge_index = tensor_to_list(train_data['ncRNA','to','Protein'].edge_label_index)
    train_edge_label = list(train_ground_truth)
    y_hat_train = list(torch.round(torch.sigmoid(torch.tensor(train_pred))).numpy())
    test_edge_index = tensor_to_list(test_data['ncRNA','to','Protein'].edge_label_index)
    test_edge_label = list(test_ground_truth)
    y_hat_test = list(torch.round(torch.sigmoid(torch.tensor(test_pred))).numpy())

    xp_train_distribution, yp_train_distribution,xr_train_distribution, yr_train_distribution = ACC_per_RP(test_edge_index,
                                                                                                           test_edge_label,
                                                                                                           y_hat_test,train_edge_index,
                                                                                                           train_edge_label,
                                                                                                           y_hat_train)
    
    # Plots for lncRNAs in terms of Number of Positive Interactions Divided by Negative Interactions
    try:
        plt.scatter(xr_train_distribution, yr_train_distribution,s=1)
        plt.xscale('log')
        plt.xlim(0.01, 100)
        plt.ylim(0, 105)
        plt.ylabel('Accuracy for each lncRNA (-)', fontname='Times New Roman', color='black')
        plt.xlabel('Number of Positive Interactions Divided by Negative Interactions (-)', fontname='Times New Roman', color='black')
        plt.title('Accuracy for each lncRNA',fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each lncRNA vs R.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    except:
        print('Error1')

    try:
        plt.hist2d(
        xr_train_distribution, 
        yr_train_distribution,
        bins=[
            np.logspace(np.log10(0.01), np.log10(100), 80),  
            np.linspace(0, 105, 80)                         
        ],
        norm=LogNorm(), 
        cmap = 'rainbow'
        )
        plt.xscale('log')
        plt.xlim(0.01, 100)
        plt.ylim(0, 105)
        plt.xlabel('Number of Positive Interactions Divided by Negative Interactions (-)',fontname='Times New Roman', color='black')
        plt.ylabel('Accuracy for each lncRNA (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each lncRNA',fontname='Times New Roman', color='black')
        cbar = plt.colorbar()
        cbar.set_label('Count', fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each lncRNA vs R, heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close()
    except:
        print('Error2')

    # Plots for Proteins in terms of Number of Positive Interactions Divided by Negative Interactions
    try:
        plt.scatter(xp_train_distribution, yp_train_distribution,s=1)
        plt.xscale('log')
        plt.xlim(0.01, 100)
        plt.ylim(0, 105)
        plt.ylabel('Accuracy for each Protein (-)',fontname='Times New Roman', color='black')
        plt.xlabel('Number of Positive Interactions Divided by Negative Interactions (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each Protein',fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each protein vs R.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    except:
        print('Error3')

    try:
        plt.hist2d(
        xp_train_distribution, 
        yp_train_distribution,
        bins=[
            np.logspace(np.log10(0.01), np.log10(100), 80),  
            np.linspace(0, 105, 80)                         
        ],
        norm=LogNorm(),
        cmap = 'rainbow'  
        )
        plt.xscale('log')
        plt.xlim(0.01, 100)
        plt.ylim(0, 105)
        plt.xlabel('Number of Positive Interactions Divided by Negative Interactions (-)',fontname='Times New Roman', color='black')
        plt.ylabel('Accuracy for each Protein (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each Protein',fontname='Times New Roman', color='black')
        cbar = plt.colorbar()
        cbar.set_label('Count', fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each protein vs R, heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close()
    except:
        print('Error4')

    xp_train_distribution, yp_train_distribution,xr_train_distribution, yr_train_distribution=ACC_per_P(test_edge_index,
                                                                                                        test_edge_label,
                                                                                                        y_hat_test,
                                                                                                        train_edge_index,
                                                                                                        train_edge_label,y_hat_train)

    # Plots for lncRNA in terms of Number of Positive Interactions
    try:
        plt.scatter(xr_train_distribution, yr_train_distribution,s=1)
        plt.xscale('log')
        plt.ylim(0, 105)
        plt.ylabel('Accuracy for each lncRNA (-)',fontname='Times New Roman', color='black')
        plt.xlabel('Number of Positive Interactions (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each lncRNA',fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each lncRNA vs No. of P Int.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    except:
        print('Error5')

    # Plots for protein in terms of Number of Positive Interactions
    try:
        plt.scatter(xp_train_distribution, yp_train_distribution,s=1)
        plt.xscale('log')
        plt.ylim(0, 105)
        plt.ylabel('Accuracy for each Protein (-)',fontname='Times New Roman', color='black')
        plt.xlabel('Number of Positive Interactions (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each Protein',fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each protein vs No. of P Int.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    except:
        print('Error6')

    try:
        plt.hist2d(
        xr_train_distribution, 
        yr_train_distribution,
        bins=[
            np.logspace(np.log10(0.01), np.log10(100), 80),  
            np.linspace(0, 105, 80)                 
        ],
        norm=LogNorm(),
        cmap = 'rainbow'  
        )
        plt.xscale('log')
        plt.ylim(0, 105)
        plt.xlim(1, 100)
        plt.xlabel('Number of Positive Interactions (-)',fontname='Times New Roman', color='black')
        plt.ylabel('Accuracy for each lncRNA (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each lncRNA',fontname='Times New Roman', color='black')
        cbar = plt.colorbar()
        cbar.set_label('Count', fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each lncRNA vs No. of P Int, heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close()
    except:
        print('Error7')

    try:
        plt.hist2d(
            xp_train_distribution, 
            yp_train_distribution,
            bins=[
                np.logspace(np.log10(0.01), np.log10(100), 80),  
                np.linspace(0, 105, 80)                      
            ],
            norm=LogNorm(),
            cmap = 'rainbow'  # optional: log color scale for large dynamic range
        )
        plt.xscale('log')
        plt.ylim(0, 105)
        plt.xlim(1, 10000)
        plt.xlabel('Number of Positive Interactions (-)',fontname='Times New Roman', color='black')
        plt.ylabel('Accuracy for each Protein (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each Protein',fontname='Times New Roman', color='black')
        cbar = plt.colorbar()
        cbar.set_label('Count', fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each protein vs No. of P Int, heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close()
    except:
        print('Error8')

    xp_train_distribution, yp_train_distribution,xr_train_distribution, yr_train_distribution=ACC_per_N(test_edge_index,
                                                                                                        test_edge_label,
                                                                                                        y_hat_test,
                                                                                                        train_edge_index,
                                                                                                        train_edge_label,
                                                                                                        y_hat_train)
    
    # Plots for lncRNA in terms of Number of Negative Interactions
    try:
        plt.scatter(xr_train_distribution, yr_train_distribution,s=1)
        plt.xscale('log')
        plt.ylim(0, 105)
        plt.ylabel('Accuracy for each lncRNA (-)',fontname='Times New Roman', color='black')
        plt.xlabel('Number of Negative Interactions (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each lncRNA',fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each lncRNA vs No. of N Int.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    except:
        print('Error9')

    # Plots for Protein in terms of Number of Negative Interactions
    try:
        plt.scatter(xp_train_distribution, yp_train_distribution,s=1)
        plt.xscale('log')
        plt.ylim(0, 105)
        plt.ylabel('Accuracy for each Protein (-)',fontname='Times New Roman', color='black')
        plt.xlabel('Number of Negative Interactions (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each Protein',fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each protein vs No. of N Int.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    except:
        print('Error10')

    try:
        plt.hist2d(
            xr_train_distribution, 
            yr_train_distribution,
            bins=[
                np.logspace(np.log10(0.01), np.log10(100), 80), 
                np.linspace(0, 105, 80)               
            ],
            norm=LogNorm(),
            cmap = 'rainbow' 
        )
        plt.xscale('log')
        plt.ylim(0, 105)
        plt.xlim(1, 100)
        plt.xlabel('Number of Negative Interactions (-)',fontname='Times New Roman', color='black')
        plt.ylabel('Accuracy for each lncRNA (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each lncRNA',fontname='Times New Roman', color='black')
        cbar = plt.colorbar()
        cbar.set_label('Count', fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each lncRNA vs No. of N Int, heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close()
    except:
        print('Error11')

    try:
        plt.hist2d(
            xp_train_distribution, 
            yp_train_distribution,
            bins=[
                np.logspace(np.log10(0.01), np.log10(100), 80),  
                np.linspace(0, 105, 80)                        
            ],
            norm=LogNorm(),
            cmap = 'rainbow' 
        )
        plt.xscale('log')
        plt.ylim(0, 105)
        plt.xlim(1, 10000)
        plt.xlabel('Number of Negative Interactions (-)',fontname='Times New Roman', color='black')
        plt.ylabel('Accuracy for each Protein (-)',fontname='Times New Roman', color='black')
        plt.title('Accuracy for each Protein',fontname='Times New Roman', color='black')
        cbar = plt.colorbar()
        cbar.set_label('Count', fontname='Times New Roman', color='black')
        save_path = os.path.join(save_folder, "Accuracy for each protein vs No. of N Int, heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close()
    except:
        print("Error12")

    # Runing model for all posible pairs and save new interations
    try:
        if args.Predicting:
            unique_tarID_dict ={}
            for i in range(len(unique_tarID)):
                key = unique_tarID['tarID'][i]
                unique_tarID_dict [key] = i

            unique_ncrnaID_to_ncrnaname_dict ={}
            for i in range(len(unique_ncID)):
                key = unique_ncID['ncID'][i]
                unique_ncrnaID_to_ncrnaname_dict [i] = key

            def pre_int_total(chosen_protein):

                chosen_protein_id = unique_tarID_dict[chosen_protein]
                chosen_protein_features = tarfeat[chosen_protein_id].unsqueeze(0)
                ncID_index = torch.arange(len(unique_ncID))
                protein_index = torch.full((len(unique_ncID),), 0)
                chosen_protein_edge_index = torch.stack([ncID_index,protein_index],dim=0)
                chosen_protein_edge_label = torch.full((len(unique_ncID),), 1)

                data2 = HeteroData()
                data2['ncRNA'].node_id = torch.arange(len(unique_ncID))
                data2['Protein'].node_id = torch.arange(1)
                data2['ncRNA'].x=ncfeat
                data2['Protein'].x=chosen_protein_features
                data2['ncRNA','Protein'].edge_label = chosen_protein_edge_label
                data2['ncRNA','Protein'].edge_index = chosen_protein_edge_index 

                data2 = T.ToUndirected()(data2)

                transform2 = T.RandomLinkSplit(
                    num_val = args.val_ratio,
                    num_test = args.test_ratio,
                    is_undirected = True,
                    key= 'edge_label',
                    add_negative_train_samples=False,
                    neg_sampling_ratio = 0,
                    edge_types=('ncRNA','to','Protein'),
                    rev_edge_types=('Protein','rev_to','ncRNA')
                    )
                train_data2, val_data2, test_data2 = transform2(data2)

                edge_label_index2 = train_data2['ncRNA','to','Protein'].edge_label_index
                edge_label2 = train_data2['ncRNA','to','Protein'].edge_label
                train_loader2 = LinkNeighborLoader(
                    data = train_data2,
                    num_neighbors=args.num_neighbors,
                    edge_label_index=(('ncRNA','to','Protein'),edge_label_index2),
                    edge_label=edge_label2,
                    batch_size = args.batch_size,
                    num_workers=8,
                    shuffle = False
                )

                model.eval()

                ncrna_list = []
                id_counter = 0
                for sampled_data in (train_loader2):
                        # continue
                    with torch.no_grad():
                        sampled_data.to(device)
                        valpred = model(sampled_data).to(float)
                        valpred = valpred.detach().cpu().numpy()
                        valpred = torch.sigmoid(torch.tensor(valpred))
                        counter2 = 0
                        for i in valpred: 
                            if i >= 0.5:
                                ncrna_name = unique_ncrnaID_to_ncrnaname_dict [id_counter + counter2]
                                ncrna_list.append(ncrna_name)
                                counter2 += 1
                                
                        id_counter += len(valpred)
                return ncrna_list

            
            interaction_dict = {}
            print('\nPredicting new interactions ...')
            for pro in tqdm.tqdm(dataset['tarID'].unique()):
                lncrna_list = pre_int_total (pro)
                interaction_dict [pro] = lncrna_list

            save_path3 = os.path.join(save_folder, "predicted_interactions.json")
            with open(save_path3,'w') as j:
                json.dump(interaction_dict,j)
            print("New interactions are saved in ",save_path3)
    except:
        print('Predicter function cannot work')
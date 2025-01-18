import torch
import pandas as pd

def datapreparation(dataset,
                rna_kmer_dict,
                protein_kmer_dict,
                rna_motif_dict,
                distance_map_dict ,
                protein_3D_dict,
                args):

    unique_ncID = dataset['ncID'].unique()
    unique_ncID = pd.DataFrame(data = 
                            {'ncID':unique_ncID,
                                'mappedncID':pd.RangeIndex(len(unique_ncID))})
    

    unique_tarID = dataset['tarID'].unique()
    unique_tarID = pd.DataFrame(data=
                                {'tarID' : unique_tarID,
                                'mappedtarID': pd.RangeIndex(len(unique_tarID))})
    

    dataset_ncID = pd.merge(dataset['ncID'],unique_ncID,left_on='ncID',right_on='ncID',how='left')
    dataset_ncID = torch.from_numpy(dataset_ncID['mappedncID'].values)

    dataset_tarID = pd.merge(dataset['tarID'],unique_tarID,left_on='tarID',right_on='tarID',how='left')
    dataset_tarID = torch.from_numpy(dataset_tarID['mappedtarID'].values)

    edge_index = torch.stack([dataset_ncID,dataset_tarID],dim=0)

    edge_label = torch.tensor(dataset['Interaction'])


    tarfeat=torch.tensor([])
    for i in range(len(unique_tarID)):

        key = unique_tarID['tarID'][i]
        feat_combined = torch.tensor(())

        if args.proteinkmer:
            for k in [5]: #Reconsider for other ks
                kmer_dict = protein_kmer_dict [f'{k}mer']
                feat_kmer = torch.tensor([kmer_dict[key]])
                feat_kmer = feat_kmer.squeeze()
                feat_combined = torch.cat((feat_combined,feat_kmer))
        else:
            feat_kmer = torch.zeros(1, 7**5) #Reconsider for other ks
            feat_kmer = feat_kmer.squeeze()
            feat_combined = torch.cat((feat_combined,feat_kmer))

        if args.distance and args.dataset == 'NPInter5':
            feat_distance_map = torch.tensor([distance_map_dict[key]])
            feat_distance_map = feat_distance_map.squeeze()
            feat_combined = torch.cat((feat_combined,feat_distance_map))
        else:
            feat_distance_map = torch.zeros(1,64)
            feat_distance_map = feat_distance_map.squeeze()
            feat_combined = torch.cat((feat_combined,feat_distance_map))

        if args.pro3D and args.dataset == 'NPInter5':
            feat3D = torch.tensor(protein_3D_dict[key]).transpose(0,1)
            feat3D = feat3D.reshape(1,3000)
            feat3D = feat3D.squeeze()
            feat_combined = torch.cat((feat_combined,feat3D))
        else:
            feat3D = torch.zeros(1,3000)
            feat3D = feat3D.squeeze()
            feat_combined = torch.cat((feat_combined,feat3D))

        feat_combined = feat_combined.unsqueeze(0)

        tarfeat=torch.cat((tarfeat,feat_combined))
    tarfeat = tarfeat.to(torch.float)


    ncfeat=torch.tensor([])

    for i in range(len(unique_ncID)):

        key = unique_ncID['ncID'][i]
        feat_combined = torch.tensor(())

        if args.lncRNAkmer:
            for k in [4,5]: #Reconsider for other ks
                kmer_dict = rna_kmer_dict [f'{k}mer']
                feat_kmer = torch.tensor([kmer_dict[key]])
                feat_kmer = feat_kmer.squeeze()
                feat_combined = torch.cat((feat_combined,feat_kmer))
        else:
            feat_kmer = torch.zeros(1,( 4**4) + (4**5)) #Reconsider for other ks
            feat_kmer = feat_kmer.squeeze()
            feat_combined = torch.cat((feat_combined,feat_kmer))

        if args.motif:
            # If we want to use motif
            feat_motif = torch.tensor([rna_motif_dict[key]])
            feat_motif = feat_motif.squeeze()
            feat_combined = torch.cat((feat_combined,feat_motif))
        else:
            feat_motif = torch.zeros(1, 1194)
            feat_motif = feat_motif.squeeze()
            feat_combined = torch.cat((feat_combined,feat_motif))
        
        feat_combined = feat_combined.unsqueeze(0)

        ncfeat=torch.cat((ncfeat,feat_combined))
    ncfeat = ncfeat.to(torch.float)


    return ncfeat, tarfeat, edge_index, edge_label, unique_ncID, unique_tarID

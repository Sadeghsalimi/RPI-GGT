import matplotlib.pyplot as plt
import torch

def tensor_to_list(tensor):
    list=[]
    for i in range(len(tensor[0])):
        rna = str(tensor[0][i].item())
        protein = str(tensor[1][i].item())
        lists=[rna,protein]
        list.append(lists)
    return(list)

def ACC_per_RP(test_edge_index ,test_edge_label,y_hat_test,train_edge_index,train_edge_label,y_hat_train):

    def accuracy_fn(y_true,y_pred):
        correct=torch.eq(y_true,y_pred).sum().item()
        acc=(correct/len(y_pred))*100
        return acc
    
    pro_distribution_dict_test = {}
    rna_distribution_dict_test = {}

    i = 0
    for pair in test_edge_index:

        rna_interaction_list = [pair[1],int(test_edge_label[i]),int(y_hat_test[i])]
        pro_interaction_list = [pair[0],int(test_edge_label[i]),int(y_hat_test[i])]

        if pair[0] in rna_distribution_dict_test:
            rna_distribution_dict_test[pair[0]].extend(rna_interaction_list)
        else:
            rna_distribution_dict_test[pair[0]] = rna_interaction_list

        if pair[1] in pro_distribution_dict_test:
            pro_distribution_dict_test[pair[1]].extend(pro_interaction_list)
        else:
            pro_distribution_dict_test[pair[1]] = pro_interaction_list

        i += 1

    pro_distribution_dict_train = {}
    rna_distribution_dict_train = {}

    i = 0
    for pair in train_edge_index:

        rna_interaction_list = [pair[1],int(train_edge_label[i]),int(y_hat_train[i])]
        pro_interaction_list = [pair[0],int(train_edge_label[i]),int(y_hat_train[i])]

        if pair[0] in rna_distribution_dict_train:
            rna_distribution_dict_train[pair[0]].extend(rna_interaction_list)
        else:
            rna_distribution_dict_train[pair[0]] = rna_interaction_list

        if pair[1] in pro_distribution_dict_train:
            pro_distribution_dict_train[pair[1]].extend(pro_interaction_list)
        else:
            pro_distribution_dict_train[pair[1]] = pro_interaction_list

        i += 1

    pro_distribution_dict = {}
    pro_fail_dict_test={}
    pro_fail_dict_train={}
    for protein in pro_distribution_dict_train.keys():
        #For Train
        num_of_interactions_train = len(pro_distribution_dict_train[protein])/3
        interaction_loc_train = 1
        positive_interaction_counter_train = 0
        while interaction_loc_train < len(pro_distribution_dict_train[protein]):
            positive_interaction_counter_train += pro_distribution_dict_train[protein][interaction_loc_train]
            interaction_loc_train += 3

        # if protein in pro_distribution_dict:
        #     pro_distribution_dict[protein].extend([num_of_interactions_train,positive_interaction_counter_train])
        # else:
        #     pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        try:
            pro_distribution_dict[protein] = [positive_interaction_counter_train/(num_of_interactions_train-positive_interaction_counter_train)]
        # pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        except:
            pro_fail_dict_train[protein] = [positive_interaction_counter_train,num_of_interactions_train]
            pro_distribution_dict[protein]=[float('nan')]        

        #For Test
        try:
            num_of_interactions_test = len(pro_distribution_dict_test[protein])/3
            interaction_loc_test = 1
            positive_interaction_counter_test = 0
            while interaction_loc_test < len(pro_distribution_dict_test[protein]):
                positive_interaction_counter_test += pro_distribution_dict_test[protein][interaction_loc_test]
                interaction_loc_test += 3

            # if protein in pro_distribution_dict:
            #     pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
            # else:
            #     pro_distribution_dict[protein] = [num_of_interactions_test,positive_interaction_counter_test]
            try:
                pro_distribution_dict[protein].extend([positive_interaction_counter_test/(num_of_interactions_test-positive_interaction_counter_test)])
            except:
                pro_fail_dict_test[protein] = [positive_interaction_counter_test,num_of_interactions_test]
                pro_distribution_dict[protein].extend([float('nan')])
            # pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
        except:
            pro_distribution_dict[protein].extend([float('nan')])

        #ACC For Train
        interaction_loc_train = 1
        y_train_list = []
        y_train_pre_list =[]
        while interaction_loc_train < len(pro_distribution_dict_train[protein]):
            y_train_list.append(pro_distribution_dict_train[protein][interaction_loc_train])
            y_train_pre_list.append(pro_distribution_dict_train[protein][interaction_loc_train+1])
            interaction_loc_train += 3
        # print(y_train_list),print(y_train_pre_list)
        Train_Accuracy=accuracy_fn(torch.tensor(y_train_list),torch.tensor(y_train_pre_list))

        #ACC For Test
        interaction_loc_test = 1
        y_test_list = []
        y_test_pre_list =[]
        # print(len(pro_distribution_dict_test[protein]))
        try:
            while interaction_loc_test < len(pro_distribution_dict_test[protein]):
                # print(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_list.append(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_pre_list.append(pro_distribution_dict_test[protein][interaction_loc_test+1])
                interaction_loc_test += 3
            # print(y_test_pre_list),print(y_test_list)
            Test_Accuracy=accuracy_fn(torch.tensor(y_test_list),torch.tensor(y_test_pre_list))
        except:
            Test_Accuracy = float('nan')

        #Adding two ACC into pro_distribution_dict
        pro_distribution_dict[protein].extend([Train_Accuracy,Test_Accuracy])

    rna_distribution_dict = {}
    fail_dict_test={}
    fail_dict_train={}
    for rna in rna_distribution_dict_train.keys():
        #For Train
        num_of_interactions_train = len(rna_distribution_dict_train[rna])/3
        interaction_loc_train = 1
        positive_interaction_counter_train = 0
        while interaction_loc_train < len(rna_distribution_dict_train[rna]):
            positive_interaction_counter_train += rna_distribution_dict_train[rna][interaction_loc_train]
            interaction_loc_train += 3

        # if protein in pro_distribution_dict:
        #     pro_distribution_dict[protein].extend([num_of_interactions_train,positive_interaction_counter_train])
        # else:
        #     pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        try:
            rna_distribution_dict[rna] = [positive_interaction_counter_train/(num_of_interactions_train - positive_interaction_counter_train)]
        except:
            fail_dict_train[rna] = [positive_interaction_counter_test,num_of_interactions_test]
            rna_distribution_dict[rna] = [float('nan')]
        # pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]

        #For Test
        try:
            num_of_interactions_test = len(rna_distribution_dict_test[rna])/3
            interaction_loc_test = 1
            positive_interaction_counter_test = 0
            while interaction_loc_test < len(rna_distribution_dict_test[rna]):
                positive_interaction_counter_test += rna_distribution_dict_test[rna][interaction_loc_test]
                interaction_loc_test += 3

            # if protein in pro_distribution_dict:
            #     pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
            # else:
            #     pro_distribution_dict[protein] = [num_of_interactions_test,positive_interaction_counter_test]

            try:
                rna_distribution_dict[rna].extend([positive_interaction_counter_test/(num_of_interactions_test - positive_interaction_counter_test)])
            except:
                fail_dict_test[rna] = [positive_interaction_counter_test,num_of_interactions_test]
                rna_distribution_dict[rna].extend([float('nan')])
            # pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
        except:
            rna_distribution_dict[rna].extend([float('nan')])

        #ACC For Train
        interaction_loc_train = 1
        y_train_list = []
        y_train_pre_list =[]
        while interaction_loc_train < len(rna_distribution_dict_train[rna]):
            y_train_list.append(rna_distribution_dict_train[rna][interaction_loc_train])
            y_train_pre_list.append(rna_distribution_dict_train[rna][interaction_loc_train+1])
            interaction_loc_train += 3
        # print(y_train_list),print(y_train_pre_list)
        Train_Accuracy=accuracy_fn(torch.tensor(y_train_list),torch.tensor(y_train_pre_list))

        #ACC For Test
        interaction_loc_test = 1
        y_test_list = []
        y_test_pre_list =[]
        # print(len(pro_distribution_dict_test[protein]))
        try:
            while interaction_loc_test < len(rna_distribution_dict_test[rna]):
                # print(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_list.append(rna_distribution_dict_test[rna][interaction_loc_test])
                y_test_pre_list.append(rna_distribution_dict_test[rna][interaction_loc_test+1])
                interaction_loc_test += 3
            # print(y_test_pre_list),print(y_test_list)
            Test_Accuracy=accuracy_fn(torch.tensor(y_test_list),torch.tensor(y_test_pre_list))
        except:
            Test_Accuracy = float('nan')

        #Adding two ACC into pro_distribution_dict
        rna_distribution_dict[rna].extend([Train_Accuracy,Test_Accuracy])

    import math
    import matplotlib.pyplot as plt

    xr_train_distribution=[]
    yr_train_distribution=[]
    xr_test_distribution=[]
    yr_test_distribution=[]
    for key,value in rna_distribution_dict.items():
        if not math.isnan(value[0]) and not math.isnan(value[2]):
            xr_train_distribution.append(value[0])
            yr_train_distribution.append(value[2])
        if not math.isnan(value[1]) and not math.isnan(value[3]):
            xr_test_distribution.append(value[1])
            yr_test_distribution.append(value[3])

    xp_train_distribution=[]
    yp_train_distribution=[]
    xp_test_distribution=[]
    yp_test_distribution=[]
    for key,value in pro_distribution_dict.items():
        xp_train_distribution.append(value[0])
        yp_train_distribution.append(value[2])
        xp_test_distribution.append(value[1])
        yp_test_distribution.append(value[3])
    
    return (xp_test_distribution, yp_test_distribution,xr_train_distribution, yr_train_distribution)

import seaborn as sns

def heatmap(x,y,title):
    im = sns.kdeplot(x=x, y=y, shade=True, cmap="YlOrRd", bw=0.4) 
    plt.xscale("log")
    im.set_xlabel('Number of Positive Interactions Divided by Negative Interactions')
    im.set_ylabel('Accuracy for each Molecule')
    im.set_title(title)
    plt.show()

def ACC_per_P (test_edge_index ,test_edge_label,y_hat_test,train_edge_index,train_edge_label,y_hat_train):

    def accuracy_fn(y_true,y_pred):
        correct=torch.eq(y_true,y_pred).sum().item()
        acc=(correct/len(y_pred))*100
        return acc
        
    pro_distribution_dict_test = {}
    rna_distribution_dict_test = {}

    i = 0
    for pair in test_edge_index:

        rna_interaction_list = [pair[1],int(test_edge_label[i]),int(y_hat_test[i])]
        pro_interaction_list = [pair[0],int(test_edge_label[i]),int(y_hat_test[i])]

        if pair[0] in rna_distribution_dict_test:
            rna_distribution_dict_test[pair[0]].extend(rna_interaction_list)
        else:
            rna_distribution_dict_test[pair[0]] = rna_interaction_list

        if pair[1] in pro_distribution_dict_test:
            pro_distribution_dict_test[pair[1]].extend(pro_interaction_list)
        else:
            pro_distribution_dict_test[pair[1]] = pro_interaction_list

        i += 1

    pro_distribution_dict_train = {}
    rna_distribution_dict_train = {}

    i = 0
    for pair in train_edge_index:

        rna_interaction_list = [pair[1],int(train_edge_label[i]),int(y_hat_train[i])]
        pro_interaction_list = [pair[0],int(train_edge_label[i]),int(y_hat_train[i])]

        if pair[0] in rna_distribution_dict_train:
            rna_distribution_dict_train[pair[0]].extend(rna_interaction_list)
        else:
            rna_distribution_dict_train[pair[0]] = rna_interaction_list

        if pair[1] in pro_distribution_dict_train:
            pro_distribution_dict_train[pair[1]].extend(pro_interaction_list)
        else:
            pro_distribution_dict_train[pair[1]] = pro_interaction_list

        i += 1


    pro_distribution_dict = {}
    pro_fail_dict_test={}
    pro_fail_dict_train={}
    for protein in pro_distribution_dict_train.keys():
        #For Train
        num_of_interactions_train = len(pro_distribution_dict_train[protein])/3
        interaction_loc_train = 1
        positive_interaction_counter_train = 0
        while interaction_loc_train < len(pro_distribution_dict_train[protein]):
            positive_interaction_counter_train += pro_distribution_dict_train[protein][interaction_loc_train]
            interaction_loc_train += 3

        # if protein in pro_distribution_dict:
        #     pro_distribution_dict[protein].extend([num_of_interactions_train,positive_interaction_counter_train])
        # else:
        #     pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        try:
            pro_distribution_dict[protein] = [positive_interaction_counter_train]
            # pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        except:
            # pro_fail_dict_train[protein] = [positive_interaction_counter_test,num_of_interactions_test]
            pro_distribution_dict[protein]=[float('nan')]  
            print("Train Error:",protein)      

        #For Test
        try:
            num_of_interactions_test = len(pro_distribution_dict_test[protein])/3
            interaction_loc_test = 1
            positive_interaction_counter_test = 0
            while interaction_loc_test < len(pro_distribution_dict_test[protein]):
                positive_interaction_counter_test += pro_distribution_dict_test[protein][interaction_loc_test]
                interaction_loc_test += 3

            # if protein in pro_distribution_dict:
            #     pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
            # else:
            #     pro_distribution_dict[protein] = [num_of_interactions_test,positive_interaction_counter_test]
            try:
                pro_distribution_dict[protein].extend([positive_interaction_counter_test])
            except:
                # pro_fail_dict_test[protein] = [positive_interaction_counter_test,num_of_interactions_test]
                pro_distribution_dict[protein].extend([float('nan')])
                print("Test Error:",protein)  
                # pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
        except:
                pro_distribution_dict[protein].extend([float('nan')])

        #ACC For Train
        interaction_loc_train = 1
        y_train_list = []
        y_train_pre_list =[]
        while interaction_loc_train < len(pro_distribution_dict_train[protein]):
            y_train_list.append(pro_distribution_dict_train[protein][interaction_loc_train])
            y_train_pre_list.append(pro_distribution_dict_train[protein][interaction_loc_train+1])
            interaction_loc_train += 3
        # print(y_train_list),print(y_train_pre_list)
        Train_Accuracy=accuracy_fn(torch.tensor(y_train_list),torch.tensor(y_train_pre_list))

        #ACC For Test
        interaction_loc_test = 1
        y_test_list = []
        y_test_pre_list =[]
            # print(len(pro_distribution_dict_test[protein]))
        try:
            while interaction_loc_test < len(pro_distribution_dict_test[protein]):
                # print(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_list.append(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_pre_list.append(pro_distribution_dict_test[protein][interaction_loc_test+1])
                interaction_loc_test += 3
            # print(y_test_pre_list),print(y_test_list)
            Test_Accuracy=accuracy_fn(torch.tensor(y_test_list),torch.tensor(y_test_pre_list))
        except:
            Test_Accuracy = float('nan')

        #Adding two ACC into pro_distribution_dict
        pro_distribution_dict[protein].extend([Train_Accuracy,Test_Accuracy])

    rna_distribution_dict = {}
    fail_dict_test={}
    fail_dict_train={}
    for rna in rna_distribution_dict_train.keys():
        #For Train
        num_of_interactions_train = len(rna_distribution_dict_train[rna])/3
        interaction_loc_train = 1
        positive_interaction_counter_train = 0
        while interaction_loc_train < len(rna_distribution_dict_train[rna]):
            positive_interaction_counter_train += rna_distribution_dict_train[rna][interaction_loc_train]
            interaction_loc_train += 3

        # if protein in pro_distribution_dict:
        #     pro_distribution_dict[protein].extend([num_of_interactions_train,positive_interaction_counter_train])
        # else:
        #     pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        try:
            rna_distribution_dict[rna] = [positive_interaction_counter_train]
        except:
            fail_dict_train[rna] = [positive_interaction_counter_test,num_of_interactions_test]
            rna_distribution_dict[rna] = [float('nan')]
        # pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]

        #For Test
        try:
            num_of_interactions_test = len(rna_distribution_dict_test[rna])/3
            interaction_loc_test = 1
            positive_interaction_counter_test = 0
            while interaction_loc_test < len(rna_distribution_dict_test[rna]):
                positive_interaction_counter_test += rna_distribution_dict_test[rna][interaction_loc_test]
                interaction_loc_test += 3

            # if protein in pro_distribution_dict:
            #     pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
            # else:
            #     pro_distribution_dict[protein] = [num_of_interactions_test,positive_interaction_counter_test]

            try:
                rna_distribution_dict[rna].extend([positive_interaction_counter_test])
            except:
                # fail_dict_test[rna] = [positive_interaction_counter_test,num_of_interactions_test]
                rna_distribution_dict[rna].extend([float('nan')])
            # pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
        except:
            rna_distribution_dict[rna].extend([float('nan')])

        #ACC For Train
        interaction_loc_train = 1
        y_train_list = []
        y_train_pre_list =[]
        while interaction_loc_train < len(rna_distribution_dict_train[rna]):
            y_train_list.append(rna_distribution_dict_train[rna][interaction_loc_train])
            y_train_pre_list.append(rna_distribution_dict_train[rna][interaction_loc_train+1])
            interaction_loc_train += 3
        # print(y_train_list),print(y_train_pre_list)
        Train_Accuracy=accuracy_fn(torch.tensor(y_train_list),torch.tensor(y_train_pre_list))

            #ACC For Test
        interaction_loc_test = 1
        y_test_list = []
        y_test_pre_list =[]
        # print(len(pro_distribution_dict_test[protein]))
        try:
            while interaction_loc_test < len(rna_distribution_dict_test[rna]):
                # print(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_list.append(rna_distribution_dict_test[rna][interaction_loc_test])
                y_test_pre_list.append(rna_distribution_dict_test[rna][interaction_loc_test+1])
                interaction_loc_test += 3
            # print(y_test_pre_list),print(y_test_list)
            Test_Accuracy=accuracy_fn(torch.tensor(y_test_list),torch.tensor(y_test_pre_list))
        except:
            Test_Accuracy = float('nan')

        #Adding two ACC into pro_distribution_dict
        rna_distribution_dict[rna].extend([Train_Accuracy,Test_Accuracy])

    import math
    import matplotlib.pyplot as plt

    xr_train_distribution=[]
    yr_train_distribution=[]
    xr_test_distribution=[]
    yr_test_distribution=[]
    for key,value in rna_distribution_dict.items():
        if not math.isnan(value[0]) and not math.isnan(value[2]):
            xr_train_distribution.append(value[0])
            yr_train_distribution.append(value[2])
        if not math.isnan(value[1]) and not math.isnan(value[3]):
            xr_test_distribution.append(value[1])
            yr_test_distribution.append(value[3])

    xp_train_distribution=[]
    yp_train_distribution=[]
    xp_test_distribution=[]
    yp_test_distribution=[]
    for key,value in pro_distribution_dict.items():
        xp_train_distribution.append(value[0])
        yp_train_distribution.append(value[2])
        xp_test_distribution.append(value[1])
        yp_test_distribution.append(value[3])
        
    return (xp_train_distribution, yp_train_distribution,xr_train_distribution, yr_train_distribution)

def ACC_per_N(test_edge_index ,test_edge_label,y_hat_test,train_edge_index,train_edge_label,y_hat_train):

    def accuracy_fn(y_true,y_pred):
        correct=torch.eq(y_true,y_pred).sum().item()
        acc=(correct/len(y_pred))*100
        return acc

    pro_distribution_dict_test = {}
    rna_distribution_dict_test = {}

    i = 0
    for pair in test_edge_index:

        rna_interaction_list = [pair[1],int(test_edge_label[i]),int(y_hat_test[i])]
        pro_interaction_list = [pair[0],int(test_edge_label[i]),int(y_hat_test[i])]

        if pair[0] in rna_distribution_dict_test:
            rna_distribution_dict_test[pair[0]].extend(rna_interaction_list)
        else:
            rna_distribution_dict_test[pair[0]] = rna_interaction_list

        if pair[1] in pro_distribution_dict_test:
            pro_distribution_dict_test[pair[1]].extend(pro_interaction_list)
        else:
            pro_distribution_dict_test[pair[1]] = pro_interaction_list

        i += 1

    pro_distribution_dict_train = {}
    rna_distribution_dict_train = {}

    i = 0
    for pair in train_edge_index:

        rna_interaction_list = [pair[1],int(train_edge_label[i]),int(y_hat_train[i])]
        pro_interaction_list = [pair[0],int(train_edge_label[i]),int(y_hat_train[i])]

        if pair[0] in rna_distribution_dict_train:
            rna_distribution_dict_train[pair[0]].extend(rna_interaction_list)
        else:
            rna_distribution_dict_train[pair[0]] = rna_interaction_list

        if pair[1] in pro_distribution_dict_train:
            pro_distribution_dict_train[pair[1]].extend(pro_interaction_list)
        else:
            pro_distribution_dict_train[pair[1]] = pro_interaction_list

        i += 1    
    
    pro_distribution_dict = {}
    pro_fail_dict_test={}
    pro_fail_dict_train={}
    for protein in pro_distribution_dict_train.keys():
        #For Train
        num_of_interactions_train = len(pro_distribution_dict_train[protein])/3
        interaction_loc_train = 1
        positive_interaction_counter_train = 0
        while interaction_loc_train < len(pro_distribution_dict_train[protein]):
            positive_interaction_counter_train += pro_distribution_dict_train[protein][interaction_loc_train]
            interaction_loc_train += 3

        # if protein in pro_distribution_dict:
        #     pro_distribution_dict[protein].extend([num_of_interactions_train,positive_interaction_counter_train])
        # else:
        #     pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        try:
            pro_distribution_dict[protein] = [num_of_interactions_train-positive_interaction_counter_train]
            # pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        except:
            # pro_fail_dict_train[protein] = [positive_interaction_counter_test,num_of_interactions_test]
            pro_distribution_dict[protein]=[float('nan')]  
            print("Train Error:",protein)      

        #For Test
        try:
            num_of_interactions_test = len(pro_distribution_dict_test[protein])/3
            interaction_loc_test = 1
            positive_interaction_counter_test = 0
            while interaction_loc_test < len(pro_distribution_dict_test[protein]):
                positive_interaction_counter_test += pro_distribution_dict_test[protein][interaction_loc_test]
                interaction_loc_test += 3

            # if protein in pro_distribution_dict:
            #     pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
            # else:
            #     pro_distribution_dict[protein] = [num_of_interactions_test,positive_interaction_counter_test]
            try:
                pro_distribution_dict[protein].extend([num_of_interactions_test-positive_interaction_counter_test])
            except:
                # pro_fail_dict_test[protein] = [positive_interaction_counter_test,num_of_interactions_test]
                pro_distribution_dict[protein].extend([float('nan')])
                print("Test Error:",protein)  
                # pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
        except:
                pro_distribution_dict[protein].extend([float('nan')])

        #ACC For Train
        interaction_loc_train = 1
        y_train_list = []
        y_train_pre_list =[]
        while interaction_loc_train < len(pro_distribution_dict_train[protein]):
            y_train_list.append(pro_distribution_dict_train[protein][interaction_loc_train])
            y_train_pre_list.append(pro_distribution_dict_train[protein][interaction_loc_train+1])
            interaction_loc_train += 3
        # print(y_train_list),print(y_train_pre_list)
        Train_Accuracy=accuracy_fn(torch.tensor(y_train_list),torch.tensor(y_train_pre_list))

        #ACC For Test
        interaction_loc_test = 1
        y_test_list = []
        y_test_pre_list =[]
            # print(len(pro_distribution_dict_test[protein]))
        try:
            while interaction_loc_test < len(pro_distribution_dict_test[protein]):
                # print(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_list.append(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_pre_list.append(pro_distribution_dict_test[protein][interaction_loc_test+1])
                interaction_loc_test += 3
            # print(y_test_pre_list),print(y_test_list)
            Test_Accuracy=accuracy_fn(torch.tensor(y_test_list),torch.tensor(y_test_pre_list))
        except:
            Test_Accuracy = float('nan')

        #Adding two ACC into pro_distribution_dict
        pro_distribution_dict[protein].extend([Train_Accuracy,Test_Accuracy])

    rna_distribution_dict = {}
    fail_dict_test={}
    fail_dict_train={}
    for rna in rna_distribution_dict_train.keys():
        #For Train
        num_of_interactions_train = len(rna_distribution_dict_train[rna])/3
        interaction_loc_train = 1
        positive_interaction_counter_train = 0
        while interaction_loc_train < len(rna_distribution_dict_train[rna]):
            positive_interaction_counter_train += rna_distribution_dict_train[rna][interaction_loc_train]
            interaction_loc_train += 3

        # if protein in pro_distribution_dict:
        #     pro_distribution_dict[protein].extend([num_of_interactions_train,positive_interaction_counter_train])
        # else:
        #     pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]
        try:
            rna_distribution_dict[rna] = [num_of_interactions_train-positive_interaction_counter_train]
        except:
            fail_dict_train[rna] = [positive_interaction_counter_test,num_of_interactions_test]
            rna_distribution_dict[rna] = [float('nan')]
        # pro_distribution_dict[protein] = [num_of_interactions_train,positive_interaction_counter_train]

        #For Test
        try:
            num_of_interactions_test = len(rna_distribution_dict_test[rna])/3
            interaction_loc_test = 1
            positive_interaction_counter_test = 0
            while interaction_loc_test < len(rna_distribution_dict_test[rna]):
                positive_interaction_counter_test += rna_distribution_dict_test[rna][interaction_loc_test]
                interaction_loc_test += 3

            # if protein in pro_distribution_dict:
            #     pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
            # else:
            #     pro_distribution_dict[protein] = [num_of_interactions_test,positive_interaction_counter_test]

            try:
                rna_distribution_dict[rna].extend([num_of_interactions_test - positive_interaction_counter_test])
            except:
                # fail_dict_test[rna] = [positive_interaction_counter_test,num_of_interactions_test]
                rna_distribution_dict[rna].extend([float('nan')])
            # pro_distribution_dict[protein].extend([num_of_interactions_test,positive_interaction_counter_test])
        except:
            rna_distribution_dict[rna].extend([float('nan')])

        #ACC For Train
        interaction_loc_train = 1
        y_train_list = []
        y_train_pre_list =[]
        while interaction_loc_train < len(rna_distribution_dict_train[rna]):
            y_train_list.append(rna_distribution_dict_train[rna][interaction_loc_train])
            y_train_pre_list.append(rna_distribution_dict_train[rna][interaction_loc_train+1])
            interaction_loc_train += 3
        # print(y_train_list),print(y_train_pre_list)
        Train_Accuracy=accuracy_fn(torch.tensor(y_train_list),torch.tensor(y_train_pre_list))

        #ACC For Test
        interaction_loc_test = 1
        y_test_list = []
        y_test_pre_list =[]
        # print(len(pro_distribution_dict_test[protein]))
        try:
            while interaction_loc_test < len(rna_distribution_dict_test[rna]):
                # print(pro_distribution_dict_test[protein][interaction_loc_test])
                y_test_list.append(rna_distribution_dict_test[rna][interaction_loc_test])
                y_test_pre_list.append(rna_distribution_dict_test[rna][interaction_loc_test+1])
                interaction_loc_test += 3
            # print(y_test_pre_list),print(y_test_list)
            Test_Accuracy=accuracy_fn(torch.tensor(y_test_list),torch.tensor(y_test_pre_list))
        except:
            Test_Accuracy = float('nan')

        #Adding two ACC into pro_distribution_dict
        rna_distribution_dict[rna].extend([Train_Accuracy,Test_Accuracy])

    import math
    import matplotlib.pyplot as plt

    xr_train_distribution=[]
    yr_train_distribution=[]
    xr_test_distribution=[]
    yr_test_distribution=[]
    for key,value in rna_distribution_dict.items():
        if not math.isnan(value[0]) and not math.isnan(value[2]):
            xr_train_distribution.append(value[0])
            yr_train_distribution.append(value[2])
        if not math.isnan(value[1]) and not math.isnan(value[3]):
            xr_test_distribution.append(value[1])
            yr_test_distribution.append(value[3])

    xp_train_distribution=[]
    yp_train_distribution=[]
    xp_test_distribution=[]
    yp_test_distribution=[]
    for key,value in pro_distribution_dict.items():
        xp_train_distribution.append(value[0])
        yp_train_distribution.append(value[2])
        xp_test_distribution.append(value[1])
        yp_test_distribution.append(value[3])
        
    return(xp_train_distribution, yp_train_distribution,xr_train_distribution, yr_train_distribution)

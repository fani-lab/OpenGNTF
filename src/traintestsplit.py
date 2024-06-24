import torch
import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling

def swap(t):
    swapped_tensor = t.clone()  # Create a copy to avoid modifying the original tensor
    swapped_tensor[0], swapped_tensor[1] = swapped_tensor[1].clone(), swapped_tensor[0].clone()
    return swapped_tensor
def split(data, dataset_name, valid_ratio=0.1, test_ratio=0.1, neg_ration=1, seed=None):
    new_train_edge_index = [[], []]
    train_edge_labels = []

    new_val_edge_index = [[], []]
    val_edge_labels = []
    val_edge_labels_index = [[], []]
    num_valid_nodes = int(valid_ratio * data['team'].num_nodes)

    new_test_edge_index = [[], []]
    test_edge_labels = []
    test_edge_labels_index = [[], []]
    num_test_nodes = int(test_ratio * data['team'].num_nodes)

    p = torch.randperm(data['team'].num_nodes)
    val_indices = p[:num_valid_nodes]
    test_indices = p[num_valid_nodes:num_valid_nodes+num_test_nodes]
    train_indices = p[num_valid_nodes+num_test_nodes:]
    old_edge_index = data[('team', 'includes', 'expert')]['edge_index']
    for i in tqdm.tqdm(range(len(old_edge_index[0]))):
        if old_edge_index[0][i] in train_indices:
            new_train_edge_index[0].append(int(old_edge_index[0][i]))
            new_train_edge_index[1].append(int(old_edge_index[1][i]))
            train_edge_labels.append(1.0)
            new_val_edge_index[0].append(int(old_edge_index[0][i]))
            new_val_edge_index[1].append(int(old_edge_index[1][i]))
            new_test_edge_index[0].append(int(old_edge_index[0][i]))
            new_test_edge_index[1].append(int(old_edge_index[1][i]))
        elif old_edge_index[0][i] in val_indices:
            val_edge_labels.append(1.0)
            val_edge_labels_index[0].append(int(old_edge_index[0][i]))
            val_edge_labels_index[1].append(int(old_edge_index[1][i]))
            new_test_edge_index[0].append(int(old_edge_index[0][i]))
            new_test_edge_index[1].append(int(old_edge_index[1][i]))


        elif old_edge_index[0][i] in test_indices:
            test_edge_labels.append(1.0)
            test_edge_labels_index[0].append(int(old_edge_index[0][i]))
            test_edge_labels_index[1].append(int(old_edge_index[1][i]))
        else:
            print('something is wrong')

    #negative sampling
    # neg_edge_index_val = negative_sampling(val_edge_labels_index, num_neg_samples=int(neg_ration * len(val_edge_labels)), method='sparse')
    # val_edge_labels_index = torch.cat([val_edge_labels_index, neg_edge_index_val], dim=1)
    # val_edge_labels = torch.cat([val_edge_labels, torch.zeros(int(neg_ration * len(val_edge_labels)))], dim=1)
    #
    # neg_edge_index_test = negative_sampling(test_edge_labels_index, num_neg_samples=int(neg_ration * len(test_edge_labels)), method='sparse')
    # test_edge_labels_index = torch.cat([test_edge_labels_index, neg_edge_index_test], dim=1)
    # test_edge_labels = torch.cat([test_edge_labels, torch.zeros(int(neg_ration * len(test_edge_labels)))], dim=1)

    # tensorization!
    new_train_edge_index = torch.tensor(new_train_edge_index)
    new_val_edge_index = torch.tensor(new_val_edge_index)
    new_test_edge_index = torch.tensor(new_test_edge_index)
    val_edge_labels_index = torch.tensor(val_edge_labels_index)
    test_edge_labels_index = torch.tensor(test_edge_labels_index)
    train_edge_label_index = new_train_edge_index
    train_edge_labels = torch.tensor(train_edge_labels)
    val_edge_labels = torch.tensor(val_edge_labels)
    test_edge_labels = torch.tensor(test_edge_labels)

    train_data = HeteroData()
    train_data['expert'].node_id = data['expert'].node_id
    train_data['skill'].node_id = data['skill'].node_id
    train_data['team'].node_id = data['team'].node_id
    train_data[('team', 'requires', 'skill')].edge_index = data[('team', 'requires', 'skill')].edge_index
    train_data[('team', 'includes', 'expert')].edge_index = new_train_edge_index
    train_data[('team', 'includes', 'expert')].edge_label = train_edge_labels
    train_data[('team', 'includes', 'expert')].edge_label_index = train_edge_label_index
    train_data[('expert', 'has', 'skill')].edge_index = data[('expert', 'has', 'skill')].edge_index
    train_data[('skill', 'rev_requires', 'team')].edge_index = data[('skill', 'rev_requires', 'team')].edge_index
    train_data[('expert', 'rev_includes', 'team')].edge_index = swap(new_train_edge_index)
    train_data[('skill', 'rev_has', 'expert')].edge_index = data[('skill', 'rev_has', 'expert')].edge_index

    val_data = HeteroData()
    val_data['expert'].node_id = data['expert'].node_id
    val_data['skill'].node_id = data['skill'].node_id
    val_data['team'].node_id = data['team'].node_id
    val_data[('team', 'requires', 'skill')].edge_index = data[('team', 'requires', 'skill')].edge_index
    val_data[('team', 'includes', 'expert')].edge_index = new_val_edge_index
    val_data[('team', 'includes', 'expert')].edge_label = val_edge_labels
    val_data[('team', 'includes', 'expert')].edge_label_index = val_edge_labels_index
    val_data[('expert', 'has', 'skill')].edge_index = data[('expert', 'has', 'skill')].edge_index
    val_data[('skill', 'rev_requires', 'team')].edge_index = data[('skill', 'rev_requires', 'team')].edge_index
    val_data[('expert', 'rev_includes', 'team')].edge_index = swap(new_val_edge_index)
    val_data[('skill', 'rev_has', 'expert')].edge_index = data[('skill', 'rev_has', 'expert')].edge_index

    test_data = HeteroData()
    test_data['expert'].node_id = data['expert'].node_id
    test_data['skill'].node_id = data['skill'].node_id
    test_data['team'].node_id = data['team'].node_id
    test_data[('team', 'requires', 'skill')].edge_index = data[('team', 'requires', 'skill')].edge_index
    test_data[('team', 'includes', 'expert')].edge_index = new_test_edge_index
    test_data[('team', 'includes', 'expert')].edge_label = test_edge_labels
    test_data[('team', 'includes', 'expert')].edge_label_index = test_edge_labels_index
    test_data[('expert', 'has', 'skill')].edge_index = data[('expert', 'has', 'skill')].edge_index
    test_data[('skill', 'rev_requires', 'team')].edge_index = data[('skill', 'rev_requires', 'team')].edge_index
    test_data[('expert', 'rev_includes', 'team')].edge_index = swap(new_test_edge_index)
    test_data[('skill', 'rev_has', 'expert')].edge_index = data[('skill', 'rev_has', 'expert')].edge_index

    torch.save(train_data, f'../output/NewSplitMethod/{dataset_name}/train.pt')
    torch.save(val_data, f'../output/NewSplitMethod/{dataset_name}/val.pt')
    torch.save(test_data, f'../output/NewSplitMethod/{dataset_name}/test.pt')

    return train_data, val_data, test_data
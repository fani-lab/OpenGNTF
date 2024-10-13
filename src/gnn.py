import itertools

import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import pytrec_eval
import numpy as np
from scipy.sparse import lil_matrix
import gc

# import all gnn models
from gs import GS
from gin import GIN
from gat import GAT
from gatv2 import GATv2
from gine import GINE

from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

# we need vecs for skill coverage
def main(vecs, data, dataset_name, epochs=25, lr=0.001, test=False, batch_size=64, full_subgraph="", graph_type="STE", dim=64, num_neighbors=None, gnn_model="gs", eval_method="sum"):
    try:
        train_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/train.fs{full_subgraph}.{graph_type}.nn{num_neighbors}.pt')
        val_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/val.fs{full_subgraph}.{graph_type}.nn{num_neighbors}.pt')
        test_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/test.fs{full_subgraph}.{graph_type}.nn{num_neighbors}.pt')
        print('splitted data loaded')
    except:
        print('splitting data')

        if graph_type == "SE":
            transform = T.RandomLinkSplit(
                disjoint_train_ratio=0.3,
                neg_sampling_ratio=5.0,  # number of negative samples per each positive sample
                add_negative_train_samples=True,
                edge_types=('expert', 'has', 'skill'),
                rev_edge_types=None,
            )
        else:
            transform = T.RandomLinkSplit(
                disjoint_train_ratio=0.3,
                neg_sampling_ratio=5.0,  # number of negative samples per each positive sample
                add_negative_train_samples=True,
                edge_types=('team', 'includes', 'expert'),
                rev_edge_types=('expert', 'rev_includes', 'team'),
            )


        train_data, val_data, test_data = transform(data)
        test_data = update_test_split(test_data, vecs, team_ratio = 0.8) # we try to manually increase the number of edges in the test split

        print("saving splitted files")
        with open(f"../output/NewSplitMethod/{dataset_name}/train.fs{0 if full_subgraph == '' else 1}.{graph_type}.nn{num_neighbors}.pt", "wb") as f:
            torch.save(train_data, f)
        with open(f"../output/NewSplitMethod/{dataset_name}/val.fs{0 if full_subgraph == '' else 1}.{graph_type}.nn{num_neighbors}.pt", "wb") as f:
            torch.save(val_data, f)
        with open(f"../output/NewSplitMethod/{dataset_name}/test.fs{0 if full_subgraph == '' else 1}.{graph_type}.nn{num_neighbors}.pt", "wb") as f:
            torch.save(test_data, f)
    
    model = Model(hidden_channels=dim, data=train_data, graph_type=graph_type, gnn_model=gnn_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f"Device: '{device}'")
    model = model.to(device)
    # train_data = train_data.to(device)
    # val_data = val_data.to(device)
    # test_data = test_data.to(device)

    if graph_type == "SE":
        edge_label_index = train_data['expert', 'has', 'skill'].edge_label_index
        edge_label = train_data['expert', 'has', 'skill'].edge_label
        edge_type = ('expert', 'has', 'skill')
    else:
        edge_label_index = train_data['team', 'includes', 'expert'].edge_label_index
        edge_label = train_data['team', 'includes', 'expert'].edge_label
        edge_type = ('team', 'includes', 'expert')

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors={key: [-1] for key in train_data.edge_types} if num_neighbors is None else num_neighbors,
        neg_sampling_ratio=5.0,
        edge_label_index=(edge_type, edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if graph_type == "SE":
        edge_label_index = val_data['expert', 'has', 'skill'].edge_label_index
        edge_label = val_data['expert', 'has', 'skill'].edge_label
        edge_type = ('expert', 'has', 'skill')

    else:
        edge_label_index = val_data['team', 'includes', 'expert'].edge_label_index
        edge_label = val_data['team', 'includes', 'expert'].edge_label
        edge_type = ('team', 'includes', 'expert')

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors={key: [-1] for key in val_data.edge_types} if num_neighbors is None else num_neighbors,
        edge_label_index=(edge_type, edge_label_index),
        edge_label=edge_label,
        batch_size= batch_size,
        shuffle=False,
    )

    if graph_type == "SE":
        edge_label_index = test_data['expert', 'has', 'skill'].edge_label_index
        edge_label = test_data['expert', 'has', 'skill'].edge_label
        edge_type = ('expert', 'has', 'skill')
    else:
        edge_label_index = test_data['team', 'includes', 'expert'].edge_label_index
        edge_label = test_data['team', 'includes', 'expert'].edge_label
        edge_type = ('team', 'includes', 'expert')

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors={key: [-1] for key in test_data.edge_types} if num_neighbors is None else num_neighbors,
        edge_label_index=(edge_type, edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(epochs):
        total_loss = total_examples = 0
        for sampled_data in tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            if graph_type == "SE":
                ground_truth = sampled_data['expert', 'has', 'skill'].edge_label
            else:
                ground_truth = sampled_data['team', 'includes', 'expert'].edge_label

            if pred.size(0) != ground_truth.size(0):
                print(f"Size mismatch: pred {pred.size()}, ground_truth {ground_truth.size()}")

            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        tqdm.write(f"Epoch: {epoch:03d}, Training Loss: {total_loss / total_examples:.4f}")

        val_preds = []
        val_ground_truths = []
        val_total_loss = val_total_examples = 0
        for sampled_data in tqdm(val_loader):
            with torch.no_grad():
                sampled_data.to(device)
                pred = model(sampled_data)
                val_preds.append(pred)
                if graph_type == "SE":
                    ground_truth = sampled_data['expert', 'has', 'skill'].edge_label
                else:
                    ground_truth = sampled_data['team', 'includes', 'expert'].edge_label

                if pred.size(0) != ground_truth.size(0):
                    print(f"Size mismatch: pred {pred.size()}, ground_truth {ground_truth.size()}")

                val_ground_truths.append(ground_truth)
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                val_total_loss += float(loss) * pred.numel()
                val_total_examples += pred.numel()
        tqdm.write(f"Epoch: {epoch:03d}, Validation Loss: {val_total_loss / val_total_examples:.4f}")
        pred = torch.cat(val_preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(val_ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        tqdm.write(f"Validation AUC Epoch {epoch}: {auc:.4f}")

    if test:
        evaluate(model, vecs, test_loader, device,
                 f"../output/NewSplitMethod/{dataset_name}/eval.{gnn_model}.e{epochs}.lr{lr}.d{dim}.nn{num_neighbors}.fs{0 if full_subgraph == '' else 1}.{graph_type}.{eval_method}.csv", full_subgraph, num_neighbors, graph_type, eval_method)
        # print(f"Test evaluation results:\n{df_mean}")

    return model

# torch.manual_seed(42) already defined seed in main.py

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_expert, x_team, edge_label_index_team_experts):
        # Ensure the indices are within bounds
        max_team_index = x_team.size(0) - 1
        max_expert_index = x_expert.size(0) - 1
        # print(f"Max team index: {max_team_index}, Max expert index: {max_expert_index}")
        # print(f"Edge label index team experts: {edge_label_index_team_experts}")

        # valid_indices = (edge_label_index_team_experts[0] <= max_team_index) & \
        #                 (edge_label_index_team_experts[1] <= max_expert_index)
        #
        # edge_label_index_team_experts = edge_label_index_team_experts[:, valid_indices]

        # Convert node embeddings to edge-level representations:
        edge_feat_team = x_team[edge_label_index_team_experts[0]]
        edge_feat_expert = x_expert[edge_label_index_team_experts[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_expert * edge_feat_team).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data, graph_type, gnn_model):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices:
        self.skill_emb = torch.nn.Embedding(data['skill'].num_nodes, hidden_channels)
        self.expert_emb = torch.nn.Embedding(data['expert'].num_nodes, hidden_channels)
        self.gnn_model = gnn_model
        if graph_type != "SE":
            self.team_emb = torch.nn.Embedding(data['team'].num_nodes, hidden_channels)
        if graph_type == "STEL":
            self.location_emb = torch.nn.Embedding(data['location'].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN
        if gnn_model == 'gs':
            self.gnn = GS(hidden_channels)
        elif gnn_model == 'gin':
            self.gnn = GIN(hidden_channels)
        elif gnn_model == 'gat':
            self.gnn = GAT(hidden_channels)
        elif gnn_model == 'gatv2':
            self.gnn = GATv2(hidden_channels)
        # elif gnn_model == 'han':
        #     self.gnn = HAN(hidden_channels)
        elif gnn_model == 'gine':
            self.gnn = GINE(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
        self.graph_type = graph_type

    def forward(self, data):
        x_dict = {
            "expert": self.expert_emb(data["expert"].node_id),
            "skill": self.skill_emb(data["skill"].node_id),
        }
        if self.graph_type != "SE":
            x_dict["team"] = self.team_emb(data["team"].node_id)
        if self.graph_type == "STEL":
            x_dict["location"] = self.location_emb(data["location"].node_id)

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        if self.gnn_model == 'gine':
            # we have to provide the set of edge attributes for gine
            self.edge_attr_dict = {}
            for edge_type in data.edge_types:
                self.edge_attr_dict[edge_type] = data[edge_type].edge_attr.view(-1, 1).float()
            x_dict = self.gnn(x_dict, data.edge_index_dict, self.edge_attr_dict)
        else:
            x_dict = self.gnn(x_dict, data.edge_index_dict)

        if self.graph_type == "SE":
            pred = self.classifier(x_dict["skill"], x_dict["expert"], data["expert", "has", "skill"].edge_label_index)
        elif self.graph_type == "STEL":
            # Add prediction logic for the STEL graph type
            pred = self.classifier(x_dict["expert"], x_dict["team"],
                                   data["team", "includes", "expert"].edge_label_index)
            # Optionally, you can add other types of prediction involving location if needed
        else:
            pred = self.classifier(x_dict["expert"], x_dict["team"],
                                   data["team", "includes", "expert"].edge_label_index)

        return pred


def create_qrel_and_run(node1_index, node2_index, predictions, ground_truth, graph_type):
    print("Sorting...")
    # Sort combined elements by predictions using numpy for efficiency
    combined_sorted = sorted(zip(node1_index, node2_index, predictions, ground_truth), key=lambda x: x[2], reverse=True)
    print("Creating qrel and run dictionaries...")

    qrel = defaultdict(dict)
    run = defaultdict(dict)

    for idx1, idx2, pred, label in tqdm(combined_sorted, desc="Processing qrels and runs"):
        label_int = int(label)  # Ensure the label is an integer
        qrel[str(idx1)][str(idx2)] = label_int
        run[str(idx1)][str(idx2)] = float(pred)  # Ensure the prediction is a float

    # Convert defaultdict back to regular dict
    return dict(qrel), dict(run)


"""

Only while creating the qrels and runs, we access the individual prediction probabilities against each team-expert pair.
For example - if e7 and e9 have been predicted probable for t5, then each of these probabilities are accessed only inside
this function

- We need a matrix like Y_ for skill coverage calculation just like the transfer learning Y_
- An ideal Y and Y_ will have the respective team_id and the corresponding list of prediction probabilities
- But an ideal Y and Y_ in the transfer learning setup does not have a team_id associated with its entry, 
because it is directly accessed from the split indices. Here we do not have such split based prediction, 
rather edge based prediction with no ordering
- So we need the respective team_id to the row of the Y and Y_
- Then we need to convert Y and Y_ to 1-hot based on the sorted prediction probabilities (Y will already have 1-hot encodings)
- Example : There are 35 experts. For team 1, the model predicts y_ value > 0.0 for 22 experts, rest (35-22 = 13) experts get 
prediction value = 0.0. Then we have in total 35 prediction values. Now for each team like this, we have 35 prediction values each
as separate rows. Now for each row, we need to sort the indices of prediction values, not the prediction values themselves.
This will be done using argsort(). Then, after having the sorted indices, suppose 22 sorted top indices of the experts, 
we will pick the top_k index based on the value of k = 2, 5 or 10. In this way, we can locate the expert indices for a every single team
- Subsequently, for each e in a team t, we will locate the skillset s and combine together against team t
- Then we compare the prediction covered skillset with the given skillset of the target team t 

"""

# create qrel and run with skills covered
def create_qrel_and_run_with_skc(vecs, node1_index, node2_index, predictions, ground_truth, graph_type):

    # node1_index => team
    # node2_index => expert

    n_teams = vecs['id'].shape[0]           # total number of teams
    n_skills = vecs['skill'].shape[1]       # total number of skills
    n_experts = vecs['member'].shape[1]     # total number of experts
    team_indices = list(set(node1_index))   # The test team indices
    team_indices_serialized = {team: serial for serial, team in enumerate(team_indices)}

    vecs['es_vecs'] = lil_matrix(np.where(np.dot(vecs['member'].transpose(), vecs['skill']).todense() > 0, 1, 0))     # Create expert-skill co-occurrence matrix
    actual_skills = vecs['skill'][team_indices].todense().astype(int)                                                 # Collecting actual skills of the test teams
    Y_ = torch.zeros((len(set(node1_index)), n_experts), dtype = float)                 # Corresponding predictions

    print("Sorting...")
    combined_sorted = sorted(zip(node1_index, node2_index, predictions, ground_truth), key=lambda x: x[2], reverse=True)    # Sort combined elements by predictions using numpy for efficiency

    print("Creating qrel and run dictionaries...")
    qrel = defaultdict(dict)
    run = defaultdict(dict)

    for idx1, idx2, pred, label in tqdm(combined_sorted, desc="Processing qrels and runs"):
        label_int = int(label)                      # Ensure the label is an integer
        qrel[str(idx1)][str(idx2)] = label_int
        run[str(idx1)][str(idx2)] = float(pred)     # Ensure the prediction is a float
        idx1_mapped = team_indices_serialized[idx1] # Map the actual index to the sorted serial of the team
        Y_[idx1_mapped][idx2] = int(label)          # Update the relevant rows with pred and labels

    # Convert defaultdict back to regular dict
    return dict(qrel), dict(run), Y_, actual_skills

def merge_predictions(skills_list, preds, method):  # method:  "sum" / "fusion"
    # Use defaultdict with float for accumulation
    predicts = {}

    # Precompute inverse for fusion method
    if method == "fusion":
        preds = {skill: {key: 1 / (60 + value) for key, value in experts.items()} for skill, experts in preds.items()}

    # Accumulate the predictions based on the method
    for skill in skills_list:
        if skill in preds:
            for expert_key, expert_prob_value in itertools.islice(preds[skill].items(), 10):
                if expert_key in predicts.keys():
                    predicts[expert_key] += expert_prob_value
                else:
                    predicts[expert_key] = expert_prob_value


    # Create a tensor for sorting
    sorted_predicts = sorted(predicts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_predicts)


def create_runs_for_SE(skills_of_teams, skills_predictions, eval_method):
    print("creating runs for Skill/Expert graph results...")
    run = {}

    for team, skills in tqdm(skills_of_teams.items(), desc="Processing teams"):
        run[team] = merge_predictions(skills_list=skills, preds=skills_predictions, method=eval_method)

    return run


def evaluate(model, vecs, test_loader, device, saving_path, full_subgraph, num_neighbors, graph_type, eval_method):
    model.eval()
    all_ground_truth = []
    all_predictions = []
    all_node1_index = []
    all_node2_index = []
    if graph_type == "SE":

        # Load data
        test_STE = torch.load(f'../output/NewSplitMethod/{saving_path.split("/")[3]}/test.fs{0 if full_subgraph == "" else 1}.STE.nn{num_neighbors}.pt')
        test_SE = torch.load(f'../output/NewSplitMethod/{saving_path.split("/")[3]}/test.fs{0 if full_subgraph == "" else 1}.SE.nn{num_neighbors}.pt')

        # Extract indices
        test_SE_experts = test_SE['expert', 'has', 'skill'].edge_index[0, :]
        test_SE_skills = test_SE['expert', 'has', 'skill'].edge_index[1, :]
        teams_skills = test_STE['team', 'requires', 'skill']['edge_index']
        test_teams_index = test_STE['team', 'includes', 'expert'].edge_index[0, :]
        test_teams_experts_index = test_STE['team', 'includes', 'expert'].edge_index[1, :]
        test_teams_label = test_STE['team', 'includes', 'expert'].edge_label

        # Initialize an empty dictionary for skills of teams
        skills_of_teams = {}

        # Populate the dictionary
        for key, value in zip(teams_skills[0, :], teams_skills[1, :]):
            key_ = str(key.item())
            if key_ in skills_of_teams:
                skills_of_teams[key_].append(str(value.item()))
            else:
                skills_of_teams[key_] = [str(value.item())]

        # Remove experts not in test_SE_experts from test_teams_experts_index
        valid_expert_mask = torch.isin(test_teams_experts_index, test_SE_experts)
        test_teams_index = test_teams_index[valid_expert_mask]
        test_teams_experts_index = test_teams_experts_index[valid_expert_mask]
        # test_teams_label = test_teams_label[valid_expert_mask]

        # Remove skills not in test_SE_skills from skills_of_teams
        valid_skills = set(test_SE_skills.numpy().astype(str))
        skills_of_teams = {team: [skill for skill in skills if skill in valid_skills]
                           for team, skills in skills_of_teams.items()}

        # Ensure no empty skill lists are kept
        skills_of_teams = {team: skills for team, skills in skills_of_teams.items() if skills}


    for i, sampled_data in enumerate(test_loader):
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            pred = torch.sigmoid(model(sampled_data)).cpu().numpy()

            if graph_type == "SE":
                ground_truth = sampled_data['expert', 'has', 'skill'].edge_label.cpu().numpy()
                node1_index = sampled_data['expert', 'has', 'skill'].edge_index[0, :].cpu().numpy()  # expert
                node2_index = sampled_data['expert', 'has', 'skill'].edge_index[1, :].cpu().numpy()  # Skill
            else:
                ground_truth = sampled_data['team', 'includes', 'expert'].edge_label.cpu().numpy()
                node1_index = sampled_data['team', 'includes', 'expert'].edge_index[0, :].cpu().numpy()  # team
                node2_index = sampled_data['team', 'includes', 'expert'].edge_index[1, :].cpu().numpy()  # expert

            all_node1_index.extend(node1_index)
            all_node2_index.extend(node2_index)
            all_predictions.extend(pred)
            all_ground_truth.extend(ground_truth)

    if graph_type == "SE":
        qrels_, runs_, Y_, actual_skills = create_qrel_and_run_with_skc(vecs, all_node2_index, all_node1_index, all_predictions, all_ground_truth,
                                            graph_type)
        qrels = {}

        for team_id_, expert_id_ in zip(test_teams_index, test_teams_experts_index):
            team_id, expert_id = str(team_id_.item()), str(expert_id_.item())
            if team_id not in qrels:
                qrels[team_id] = {}
            qrels[team_id][expert_id] = 1

        runs = create_runs_for_SE(skills_of_teams, runs_, eval_method=eval_method)

    else:
        qrels, runs, Y_, actual_skills = create_qrel_and_run_with_skc(vecs, all_node1_index, all_node2_index, all_predictions, all_ground_truth,
                                          graph_type)
    # del qrels_, runs_

    skill_coverage = calculate_skill_coverage(vecs, actual_skills, Y_, [2, 5, 10])
    aucroc = roc_auc_score(all_ground_truth, all_predictions)
    print(f'AUC-ROC: {aucroc * 100}')

    metrics_to_evaluate = {
        'P_2', 'P_5', 'P_10', 'recall_2', 'recall_5', 'recall_10',
        'ndcg_cut_2', 'ndcg_cut_5', 'ndcg_cut_10',
        'map_cut_2', 'map_cut_5', 'map_cut_10'
    }

    try:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_evaluate)
        metrics = evaluator.evaluate(runs)

        aggregated_metrics = {metric: pytrec_eval.compute_aggregated_measure(
            metric, [query_metrics[metric] for query_metrics in metrics.values()]
        ) for metric in next(iter(metrics.values())).keys()}

        aggregated_metrics['aucroc'] = aucroc
        aggregated_metrics['skc_2'] = skill_coverage['skc_2']
        aggregated_metrics['skc_5'] = skill_coverage['skc_5']
        aggregated_metrics['skc_10'] = skill_coverage['skc_10']

        for metric, score in aggregated_metrics.items():
            print(f'{metric} average: {score * 100}')

        df = pd.DataFrame([aggregated_metrics])
        df.to_csv(saving_path, index=False)
        print(f'Aggregated metrics saved to {saving_path}')

    except Exception as e:
        print(f'Error evaluating metrics: {e}')

'''

*** Only implemented for STE ***

update_test_split : Updates the existing test_split with more test edges based on some criteria

vecs : the 1-hot matrix needed to fetch the ground truth experts for the test teams
team_ratio : the ratio of the test teams to be picked from the test split, to find their corresponding `True Experts`

The primary reason to update the existing test_split is due to the necessity of predicting probabilities
against all possible experts for each team. Because we filter out the top k probabilities from all expert probabilities 
for a team, we need to have that many labeled edges in the test split. Currently this huge number of labeled edges 
(num_test_teams * total number of experts) are not possible, because that will force the model to predict for that many edges 
throughout the test phase. So, we now take the unique test team indices, then fetch the corresponding ground truth 
expert 1-hot vectors from `vecs`, and then modify the existing edge_label_index of the test_split with the ground truths and
remaining number of negative edges (the edges that do not exist). Now the total size of the prediction edges will be =  
(max possible (m) true experts found for a single team in the test split + another `m` experts which do not exist for each team)

'''
def update_test_split(test_split, vecs, team_ratio = 0.8):

    import numpy as np
    import math

    test_team_indices = list(set(np.asarray(test_split['team', 'includes', 'expert'].edge_label_index[0]))) # indices of the test teams in the test split edge_label_index
    test_team_indices_trimmed = np.sort(np.random.choice(test_team_indices, size = int(math.floor(len(test_team_indices) * 0.8)), replace = False)) # subset of the actual indices selected to be the new test teams
    test_team_experts = np.asarray(vecs['member'][test_team_indices_trimmed].todense()) # corresponding experts of the team indices from above
    num_test_teams = test_team_indices_trimmed.shape[0] # total number of test teams
    num_max_experts = int(np.max(np.sum(test_team_experts, axis = 1))) # max possible (m) true experts found for a single team in the test split
    num_total_edges = num_max_experts * 2  # actual maximum experts + equal number of negative samples (or more depending on the deficit of true labels)

    edge_label_index = [[], []]  # To store the rows: Team nodes and Expert nodes
    edge_label = []  # To store the edge labels: 1 for true edges, 0 for false edges

    for row_idx, team_idx in enumerate(test_team_indices_trimmed): # The test_team_experts matrix is sorted based on the test_team_indices, so we can access them by row_idx
        # Get indices of true edges (where the value is 1)
        true_expert_indices = np.where(test_team_experts[row_idx] == 1)[0] # The test team matrix is arranged in the ascending order of the team indices
        num_true_edges = len(true_expert_indices)

        # Limit the number of true edges to max_experts
        num_true_edges_to_add = min(num_true_edges, num_max_experts) # This can be less than the max number of possible experts for a single team
        selected_true_experts = true_expert_indices[:num_true_edges_to_add]

        # Add the true edges to edge_label_index and labels
        edge_label_index[0].extend([team_idx] * num_true_edges_to_add)
        edge_label_index[1].extend(selected_true_experts)
        edge_label.extend([1] * num_true_edges_to_add)

        # Get indices of false edges (where the value is 0)
        false_expert_indices = np.where(test_team_experts[row_idx] == 0)[0]
        num_false_edges_to_add = abs(num_true_edges_to_add - num_max_experts) + num_max_experts # Based on the number of already added true edges

        # Randomly sample false edges (experts with 0 value)
        selected_false_experts = np.random.choice(false_expert_indices, num_false_edges_to_add, replace=False)

        # Add the false edges to edge_label_index and labels
        edge_label_index[0].extend([team_idx] * num_false_edges_to_add)
        edge_label_index[1].extend(selected_false_experts)
        edge_label.extend([0] * num_false_edges_to_add)

        # Convert edge_label_index to a tensor for PyG compatibility
    edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
    edge_label = torch.tensor(edge_label, dtype=torch.float)

    test_split['team','includes','expert'].edge_label_index = edge_label_index
    test_split['team','includes','expert'].edge_label = edge_label
    test_split.validate(raise_on_error = True)

    return test_split


# calculate skill_coverage for k = [2, 5, 10] for example
def calculate_skill_coverage(vecs, actual_skills, Y_, top_k):

    print(f"Calculating Skill Coverage for {Y_.shape[0]} predictions")

    if not isinstance(vecs['es_vecs'], np.ndarray):
        vecs['es_vecs'] = np.where(np.asarray(vecs['es_vecs'].todense()) > 0, 1, 0)
    skill_coverage = {}
    top_k_y_ = convert_to_one_hot(Y_, top_k) # convert the predicted experts to one-hot encodings based on top-k recommendations

    # we have to calculate skill_coverage for each value in the list top_k (2, 5 and 10 for example)
    for k in top_k:
        print(f"---- Calculating skc for k = {k}")
        Y_ = top_k_y_[k] # the 1-hot converted matrix for top k recommendations

        predicted_skills = np.where(np.dot(Y_, vecs['es_vecs']).astype(int) > 0, 1, 0)                                  # skill occurrence matrix of predicted members of shape (1 * |s|) for each row
        skills_overlap = ((predicted_skills & actual_skills) > 0).astype(int)                                           # overlap of skills in each row between predicted and actual
        skill_coverage[f'skc_{k}'] = np.average([r1.sum()/r2.sum() for r1,r2 in zip(skills_overlap,actual_skills)])     # avg coverage over all the predicted rows
        print(f"---- Calculated skc for k = {k}")

    return skill_coverage

# convert the top k expert prediction probabilities into 1-hot occurrences
# here top_k is a list of k's
def convert_to_one_hot(y_, top_k):
    print("Converting the prediction probabilities to 1-hot predictions")
    top_k_matrices = {}

    for k in top_k:
        print(f"-------- Converting for k = {k}")
        result = np.zeros_like(y_)

        for i in tqdm(range(y_.shape[0])):
            top_k_indices = np.argsort(y_[i])[-k:] # get the indices of the top k values
            result[i, top_k_indices] = 1 # set the top k values to 1

        top_k_matrices[k] = result
        print(f"-------- Converted for k = {k}\n")

    return top_k_matrices # |test_instances| * |num_test_instance_experts| for each k in top_k
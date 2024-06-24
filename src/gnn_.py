
def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode.
    test_preds = []
    test_ground_truths = []
    # Iterate over batches of data from the test loader.
    for sampled_data in test_loader:
        with torch.no_grad():  # Disable gradient calculation for testing.
            sampled_data.to(device)  # Move the batch of data to the specified device.
            pred = model(sampled_data)  # Get predictions from the model.
            test_preds.append(pred)  # Store the predictions.
            ground_truth = sampled_data['team', 'includes', 'expert'].edge_label  # Get the ground truth labels.
            test_ground_truths.append(ground_truth)  # Store the ground truth labels.
    # Concatenate all predictions and ground truth labels.
    pred = torch.cat(test_preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(test_ground_truths, dim=0).cpu().numpy()
    return pred, ground_truth


def create_qrel_and_run(team_index, expert_index, predictions, ground_truth):
    print("Sorting...")
    combined = list(zip(team_index, expert_index, predictions, ground_truth))

    # Sort by the second element (assuming you want to sort by predictions)
    combined_sorted = sorted(combined, key=lambda x: x[2])
    print("Creating qrel and run dictionaries...")

    qrel = defaultdict(dict)
    run = defaultdict(dict)

    for team_idx, expert_idx, pred, label in combined_sorted:
        label_int = int(label)  # Ensure the label is an integer
        qrel[str(team_idx)][str(expert_idx)] = label_int
        run[str(team_idx)][str(expert_idx)] = float(pred)  # Ensure the prediction is a float

    # Convert defaultdict back to regular dict
    qrel = dict(qrel)
    run = dict(run)

    return qrel, run


def evaluate(model, test_loader, device, saving_path):
    model.eval()
    all_ground_truth = []
    all_predictions = []
    all_team_index = []
    all_expert_index = []

    for i, sampled_data in enumerate(test_loader):
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            pred = torch.sigmoid(model(sampled_data)).cpu().numpy()
            ground_truth = sampled_data['team', 'includes', 'expert'].edge_label.cpu().numpy()
            all_team_index.extend(sampled_data['team', 'includes', 'expert'].edge_index[0, :].cpu().numpy())
            all_expert_index.extend(sampled_data['team', 'includes', 'expert'].edge_index[1, :].cpu().numpy())
            all_predictions.extend(pred)
            all_ground_truth.extend(ground_truth)

    qrels, runs = create_qrel_and_run(all_team_index, all_expert_index, all_predictions, all_ground_truth)

    aucroc = roc_auc_score(all_ground_truth, all_predictions)
    print(f'AUC-ROC: {aucroc}')

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
        for metric, score in aggregated_metrics.items():
            print(f'{metric} average: {score}')
        df = pd.DataFrame([aggregated_metrics])
        df.to_csv(saving_path, index=False)
        print(f'Aggregated metrics saved to {saving_path}')
    except Exception as e:
        print(f'Error evaluating metrics: {e}')


def main(data, dataset_name, epochs=25, lr=0.001, test=False, full_subgraph="", graph_type="STE"):
    try:
        train_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/train.pt')
        val_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/val.pt')
        test_data = torch.load(f'../output/NewSplitMethod/{dataset_name}/test.pt')
        print('splitted data loaded')
    except:
        print('splitting data')
        transform = T(
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=50.0,  # number of negative samples per each positive sample
            add_negative_train_samples=True,
            edge_types=('team', 'includes', 'expert'),
            rev_edge_types=('expert', 'rev_includes', 'team'),
        )
        train_data, val_data, test_data = transform(data)
        print("saving splitted files")
        with open(f"../output/NewSplitMethod/{dataset_name}/train.pt", "wb") as f:
            torch.save(train_data, f)
        with open(f"../output/NewSplitMethod/{dataset_name}/val.pt", "wb") as f:
            torch.save(val_data, f)
        with open(f"../output/NewSplitMethod/{dataset_name}/test.pt", "wb") as f:
            torch.save(test_data, f)

    model = Model(hidden_channels=4, data=train_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)

    edge_label_index = train_data['team', 'includes', 'expert'].edge_label_index
    edge_label = train_data['team', 'includes', 'expert'].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors={key: [-1] for key in train_data.edge_types},
        neg_sampling_ratio=2.0,
        edge_label_index=(('team', 'includes', 'expert'), edge_label_index),
        edge_label=edge_label,
        batch_size=64,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    edge_label_index = val_data['team', 'includes', 'expert'].edge_label_index
    edge_label = val_data['team', 'includes', 'expert'].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors={key: [-1] for key in train_data.edge_types},
        edge_label_index=(('team', 'includes', 'expert'), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * 64,
        shuffle=False,
    )

    edge_label_index = test_data['team', 'includes', 'expert'].edge_label_index
    edge_label = test_data['team', 'includes', 'expert'].edge_label
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors={key: [-1] for key in train_data.edge_types},
        edge_label_index=(('team', 'includes', 'expert'), edge_label_index),
        edge_label=edge_label,
        batch_size=1024,
        shuffle=False,
    )

    for epoch in range(epochs):
        total_loss = total_examples = 0
        for sampled_data in tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data['team', 'includes', 'expert'].edge_label
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
                ground_truth = sampled_data['team', 'includes', 'expert'].edge_label
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
        evaluate(model, test_loader, device, f"../output/NewSplitMethod/{dataset_name}/eval_e{epochs}_lr{lr}{full_subgraph}.csv")
        # print(f"Test evaluation results:\n{df_mean}")

    return model


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_expert, x_team, edge_label_index_team_experts):
        # Convert node embeddings to edge-level representations:
        edge_feat_team = x_team[edge_label_index_team_experts[0]]
        edge_feat_expert = x_expert[edge_label_index_team_experts[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_expert * edge_feat_team).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices:
        self.skill_emb = torch.nn.Embedding(data['skill'].num_nodes, hidden_channels)
        self.expert_emb = torch.nn.Embedding(data['expert'].num_nodes, hidden_channels)
        self.team_emb = torch.nn.Embedding(data['team'].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        # self.gnn = to_hetero(self.gnn, metadata=meta)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data):
        x_dict = {
            "expert": self.expert_emb(data["expert"].node_id),
            "skill": self.skill_emb(data["skill"].node_id),
            "team": self.team_emb(data["team"].node_id)
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(x_dict["expert"], x_dict["team"], data["team", "includes", "expert"].edge_label_index)
        return pred

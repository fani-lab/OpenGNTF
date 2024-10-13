"""

This file is for experimentation purpose

We try to load existing model and data files and calculate the metrics.
Subsequently we save them in the required csv files.
This file was created to calculate skill coverage from existing model files
and then update the relevant csv files

"""


from gnn import evaluate
from torch_geometric.loader import LinkNeighborLoader

import pandas as pd
import torch
import argparse
import pickle
torch.manual_seed(0)

# This method is created for experimentation purposes only
def eval(params):

    # Initializers
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_subgraph = params['full_subgraph']     # 0 or 1
    dataset_pth = params['data_path'][0]        # Considering only the first path for now
    dataset_name = dataset_pth.split('/')[-2]   # dblp, imdb, imdb-toy etc.
    epochs = params['epoch']
    dim = params['dim']
    graph_type = params['graph_type']
    gnn_model = params['gnn_model']
    eval_method = params['eval_method']         # sum or fusion
    num_neighbors = params['num_neighbors']     # [20], [20, 10], [30, 20] etc.
    lr = params['lr']                           # Learning rate of the GNN model


    # The files to load and save
    model_filename = f'../output/NewSplitMethod' + '/' + dataset_pth.split('/')[-2] + f'/model.{gnn_model}.e{epochs}.lr{lr}.d{dim}.nn{num_neighbors}.fs{full_subgraph}.{graph_type}.{eval_method}.pt'
    test_data_filename = f"../output/NewSplitMethod/{dataset_name}/test.fs{full_subgraph}.{graph_type}.nn{num_neighbors}.pt"
    output_filename = f"../output/NewSplitMethod/{dataset_name}/eval.{gnn_model}.e{epochs}.lr{lr}.d{dim}.nn{num_neighbors}.fs{full_subgraph}.{graph_type}.{eval_method}.csv"

    # Load model file
    try :
        model = torch.load(model_filename)
    except FileNotFoundError:
        print(f"Model file : {model_filename} not found!")
        return 1

    # Load test data
    try:
        test_data = torch.load(test_data_filename)
    except FileNotFoundError:
        print(f"Test Data file : {test_data_filename} not found!")
        return 1

    # Load vecs file
    try:
        with open(f'../data/{dataset_name}/teamsvecs.pkl', 'rb') as f: vecs = pickle.load(f)
    except FileNotFoundError:
        print(f"Teamsvecs file : ../data/{dataset_name}/teamsvecs.pkl not found!")
        return 1


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
        batch_size=params['batch_size'],
        shuffle=False,
    )

    evaluate(model, vecs, test_loader, device, output_filename, full_subgraph, num_neighbors, graph_type, eval_method)


if __name__ == '__main__':
    params = {
        "data_path": [
            # "../data/imdb/",
            "../data/dblp/",
        ],
        "epoch": 10, # initially it was 10
        "lr": 0.001,
        "batch_size": 128, # initially it was 1024
        "num_neighbors" : None,
        "dim" : 64, # initially 64
        "graph_type": "STE",  # STE -> Skill/Team/Expert, SE -> Skill/Expert
        "full_subgraph": 0,  # 1 -> complete subgraph, 0 -> non-complete subgraph
        "eval_method": "fusion",  # "sum" -> normal, "fusion" -> 1/(60+x)
        "gnn_model" : "gs", # gs, gin, gat, gatv2, han, gine
    }

    # Create the parser
    parser = argparse.ArgumentParser(description="Optional arguments for the script")

    # Add optional arguments
    parser.add_argument('--epoch', type=int, help='Number of epochs to train')
    parser.add_argument('--data_path', type=str, nargs ='+', help='Data to use')
    parser.add_argument('--gnn_model', type=str, help='GNN model to use')
    parser.add_argument('--graph_type', type=str, help='Type of graph')
    parser.add_argument('--dim', type=int, help='dimension of hidden layers')
    parser.add_argument('--num_neighbors', type=int, nargs='+', help='Number of neighbors for subgraph sampling')
    parser.add_argument('--full_subgraph', type=int, choices=[0, 1], help='Whether to use full subgraph')
    parser.add_argument('--eval_method', type=str, help='Evaluation method')

    # Parse the arguments
    args = parser.parse_args()

    # Update default_parameters with provided arguments if they exist
    if args.epoch:
        params['epoch'] = args.epoch
    if args.data_path:
        params['data_path'] = args.data_path
    if args.gnn_model:
        params['gnn_model'] = args.gnn_model
    if args.graph_type:
        params['graph_type'] = args.graph_type
    if args.dim:
        params['dim'] = args.dim
    if args.num_neighbors:
        params['num_neighbors'] = args.num_neighbors
    if args.full_subgraph is not None:
        params['full_subgraph'] = args.full_subgraph
    if args.eval_method:
        params['eval_method'] = args.eval_method

    eval(params)
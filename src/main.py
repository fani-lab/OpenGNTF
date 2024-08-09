import dataPreparation
import gnn
import glob
from ast import literal_eval
import pandas as pd
import torch
import argparse
torch.manual_seed(0)

def main(params: dict):
    subgraph = "-full_subgraph" if params["full_subgraph"] == 1 else ""
    fs = params["full_subgraph"]    # a small token identifier for output files; 1 = full_subgraph used
    graph_typ = params["graph_type"]
    gnn_model = params["gnn_model"]
    eval_method = params["eval_method"]

    for dataset_pth in params["data_path"]:
        print(f'dataset path = {dataset_pth}')
        try:
            if params["full_subgraph"] == 1:
                if graph_typ == "SE":
                    data = torch.load(dataset_pth + 'data-full_subgraph-SE.pt')
                    print('data - complete subgraph for SE graph loaded')
                elif graph_typ == "STE":
                    data = torch.load(dataset_pth + 'data-full_subgraph-STE.pt')
                    print('data - complete subgraph for STE graph loaded')
                elif graph_typ == "STEL":
                    data = torch.load(dataset_pth + 'data-full_subgraph-STEL.pt')
                    print('data - complete subgraph for STEL graph loaded')
            else:
                if graph_typ == "SE":
                    data = torch.load(dataset_pth + 'data-SE.pt')
                    print('data - non-complete subgraph for SE graph loaded')
                elif graph_typ == "STE":
                    data = torch.load(dataset_pth + 'data-STE.pt')
                    print('data - non-complete for STE graph loaded')
                elif graph_typ == "STEL":
                    data = torch.load(dataset_pth + 'data-STEL.pt')
                    print('data - non-complete for STEL graph loaded')

        except FileNotFoundError:
            print("preparing data ...")
            try:
                experts_df = pd.read_pickle(dataset_pth + "experts.pkl")
                print("experts pkl found!")
            except FileNotFoundError:
                print("experts pkl not found!")
                experts_df = dataPreparation.experts_df_from_teamsvec(pd.read_pickle(dataset_pth + "teamsvecs.pkl"))
                pd.to_pickle(experts_df, dataset_pth + "experts.pkl")
                print("experts pkl saved!")
            try:
                teams_df = pd.read_pickle(dataset_pth + "teams_sorted.pkl")
                print("teams pkl found!")
            except FileNotFoundError:
                print("teams pkl not found!")
                teams_df = dataPreparation.teams_df_from_teamsvec(pd.read_pickle(dataset_pth + "teamsvecs.pkl"))
                pd.to_pickle(teams_df, dataset_pth + "teams_sorted.pkl")
                print("teams pkl saved!")
            data = dataPreparation.main(experts_df, teams_df, path=dataset_pth + f'/data{subgraph}-{graph_typ}.pt',
                                        full_subgraph=subgraph, graph_type=graph_typ)
            print('data saved')

        final_model = gnn.main(data, dataset_pth.split('/')[-2], epochs=params["epoch"], lr=params["lr"],
                               batch_size=params["batch_size"], test=True, full_subgraph=subgraph, graph_type=graph_typ, gnn_model=gnn_model,
                               eval_method=eval_method)

        torch.save(final_model, '../output/NewSplitMethod' + '/' + dataset_pth.split('/')[-2] + f'/model_{gnn_model}_e{params["epoch"]}_lr{params["lr"]}_fs{fs}_{graph_typ}_{eval_method}.pt')


# sample argument based command
# python -u main.py --data_path ../data/dblp/ --gnn_model gin --graph_type STE --full_subgraph 0 --eval_method fusion
# this will output model and csv files as
# model_gin_e10_lr0.001_fs0_STE_fusion.pt
# eval_gin_e10_lr0.001_fs0_STE_fusion.pt

if __name__ == '__main__':
    parameters = {
        "data_path": [
            # "../data/imdb/",
            "../data/dblp/",
        ],
        "epoch": 10,
        "lr": 0.001,
        "batch_size": 1024,
        "graph_type": "STE",  # STE -> Skill/Team/Expert, SE -> Skill/Expert
        "full_subgraph": 0,  # 1 -> complete subgraph, 0 -> non-complete subgraph
        "eval_method": "sum",  # "sum" -> normal, "fusion" -> 1/(60+x)
        "gnn_model" : "gs", # gs, gin, gat, gatv2, han, gine
    }

    # Create the parser
    parser = argparse.ArgumentParser(description="Optional arguments for the script")

    # Add optional arguments
    parser.add_argument('--data_path', type=str, nargs ='+', help='Data to use')
    parser.add_argument('--gnn_model', type=str, help='GNN model to use')
    parser.add_argument('--graph_type', type=str, help='Type of graph')
    parser.add_argument('--full_subgraph', type=int, choices=[0, 1], help='Whether to use full subgraph')
    parser.add_argument('--eval_method', type=str, help='Evaluation method')

    # Parse the arguments
    args = parser.parse_args()

    # Update default_parameters with provided arguments if they exist
    if args.data_path:
        parameters['data_path'] = args.data_path
    if args.gnn_model:
        parameters['gnn_model'] = args.gnn_model
    if args.graph_type:
        parameters['graph_type'] = args.graph_type
    if args.full_subgraph is not None:
        parameters['full_subgraph'] = args.full_subgraph
    if args.eval_method:
        parameters['eval_method'] = args.eval_method

    main(parameters)

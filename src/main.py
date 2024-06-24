import dataPreparation
import gnn
import glob
from ast import literal_eval
import pandas as pd
import torch


def main(params: dict):
    subgraph = "-full_subgraph" if params["full_subgraph"] == 1 else ""
    graph_typ = params["graph_type"]
    for dataset_pth in params["data_path"]:
        try:
            if params["full_subgraph"] == 1:
                if graph_typ == "SE":
                    data = torch.load(dataset_pth + 'data-full_subgraph-SE.pt')
                    print('data - complete subgraph for SE graph loaded')
                elif graph_typ == "STE":
                    data = torch.load(dataset_pth + 'data-full_subgraph-STE.pt')
                    print('data - complete subgraph for STE graph loaded')
            else:
                if graph_typ == "SE":
                    data = torch.load(dataset_pth + 'data-SE.pt')
                    print('data - non-complete subgraph for SE graph loaded')
                elif graph_typ == "STE":
                    data = torch.load(dataset_pth + 'data-STE.pt')
                    print('data - non-complete for STE graph loaded')

        except FileNotFoundError:
            print("preparing data ...")
            try:
                experts_df = pd.read_pickle(dataset_pth+"experts.pkl")
                print("experts pkl found!")
            except FileNotFoundError:
                print("experts pkl not found!")
                experts_df = dataPreparation.experts_df_from_teamsvec(pd.read_pickle(dataset_pth + "teamsvecs.pkl"))
                pd.to_pickle(experts_df, dataset_pth+"experts.pkl")
                print("experts pkl saved!")
            try:
                teams_df = pd.read_pickle(dataset_pth+"teams_sorted.pkl")
                print("teams pkl found!")
            except FileNotFoundError:
                print("teams pkl not found!")
                teams_df = dataPreparation.teams_df_from_teamsvec(pd.read_pickle(dataset_pth + "teamsvecs.pkl"))
                pd.to_pickle(teams_df,  dataset_pth + "teams_sorted.pkl")
                print("teams pkl saved!")
            data = dataPreparation.main(experts_df, teams_df, path=dataset_pth + f'/data{subgraph}-{graph_typ}.pt', full_subgraph=subgraph, graph_type=graph_typ)
            print('data saved')

        final_model = gnn.main(data, dataset_pth.split('/')[-2], epochs=10, lr=0.001, test=True, full_subgraph=subgraph, graph_type=graph_typ)
        torch.save(final_model, '../output/NewSplitMethod' + '/' + dataset_pth.split('/')[-2] + '/model.pt')


if __name__ == '__main__':
    parameters = {
        "data_path": [
                       "../data/imdb/",
                      # "../data/dblp/",
                      ],
        "graph_type": "SE",  # STE -> Skill/Team/Expert, SE -> Skill/Expert
        "full_subgraph": 0,   # 1 -> complete subgraph, 0 -> non-complete subgraph

    }
    main(parameters)
# Graph Neural Team Recommendation: Toward an Integrated End-2-End Approach

Team recommendation methods have employed graph neural networks in recommending teams of experts to do a task at hand, given an input required skills. However, they suffer from: 
>> The skill embeddings are pretrained `disjointedly` by a heterogeneous graph neural network on an expert collaboration graph, so, preventing end-to-end optimization for effective yet efficient prediction of the optimal subset of experts.

>> Due to the large pool of experts in real-world scenarios, the feedforward classifier is adversely affected by the `high-dimensional multi-hot vector` in the output layer, where each label corresponds to an expert.

In this project, we reformulate the team recommendation problem into an end-to-end `link prediction` task between expert nodes and the nodes for the required subset of skills in the expert collaboration graph, 
>> Omittng the unnecessary complexities of a disjoint two-phase training procedure

>> Addressing the curse of output sparsity arising from large pools of experts.

Our experiments on large-scale datasets from various domains showcase the superiority of our proposed approach. See [Results](#Results) below.


OpenGNTG focuses on training and evaluating Graph Neural Networks (GNNs) using subgraphs of different types (e.g., Skill/Team/Expert (STE), Skill/Expert (SE), and STEL). The pipeline processes datasets, prepares subgraph data, and trains specified GNN models with user-defined parameters.

![image](https://github.com/user-attachments/assets/4913d39e-1120-4182-b063-326c05aace8e)


## Features

- **Subgraph Preparation**: Creates complete or non-complete subgraphs for different graph types.
- **Multiple GNN Models**: Supports various GNN models like GIN, GS, GAT, GATv2, HAN, and GINE.
- **Customizable Parameters**: Allows users to configure training parameters such as epochs, learning rate, batch size, and subgraph type through command-line arguments.
- **Data Preparation**: Automatically processes data files and handles missing data by generating required data from the source files.
- **Evaluation**: Supports different evaluation methods including normal and fusion-based evaluation.

## Project Structure
```
.
├── src/
│   ├── main.py              # Main script for training and evaluation
│   ├── dataPreparation.py   # Contains functions for preparing the graph data
│   ├── gnn.py               # Contains functions for training and evaluating GNN models
├── data/                    # Directory of datasets
|   ├── imdb/
|   ├── dblp/
|   ├── toy-dataset/
└── output/                  # Directory of outputs
```

## Prerequisites

- Python 3.8+
- Required Python packages (listed in `requirements.txt`):


## Installation

1. Clone the repository:

   ```bash
   git clone [https://github.com/fani-lab/OpenGNTF.git](https://github.com/fani-lab/OpenGNTF.git)
   cd OpenGNTF
   ```

2. Install the required packages (creating a virtual environment is highly recommended):

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the `main.py` script with optional parameters for training the GNN model:

```bash
python -u main.py --data_path ../data/imdb/ --gnn_model gin --graph_type STE --full_subgraph 0 --eval_method fusion
```

### Arguments

- `--data_path` (str, list): Paths to the datasets to be used for training.
- `--gnn_model` (str): The GNN model to use. Options include `gs`, `gin`, `gat`, `gatv2`, `han`, and `gine`.
- `--graph_type` (str): The type of graph structure (`SE`, `STE`, `STEL`).
- `--epoch` (int): Number of training epochs.
- `--dim` (int): Dimension of hidden layers.
- `--num_neighbors` (int, list): Number of neighbors for subgraph sampling.
- `--full_subgraph` (int): Use `1` for a complete subgraph, `0` for a non-complete subgraph.
- `--eval_method` (str): Evaluation method (`fusion` or `sum`).

### Example

```bash
python -u main.py --data_path ../data/dblp/ --gnn_model gs --graph_type STE --full_subgraph 0 --epoch 20 --lr 0.001 --batch_size 128 --dim 64 --eval_method fusion
```

This example trains a GNN using the "gs" model on the "STE" graph type, without using a full subgraph, for 20 epochs, with a learning rate of 0.001 and a batch size of 128.

## Output

The trained model and evaluation results are saved in the `output/` directory with the following naming convention:

```
model.<gnn_model>.e<epoch>.lr<learning_rate>.d<dim>.nn<num_neighbors>.fs<full_subgraph>.<graph_type>.<eval_method>.pt
```

For example:

```
model_gs_e20_lr0.001_d64_nnNone_fs0_STE_fusion.pt
```

## Data Preparation

The script expects the following data files in each dataset folder:

- `experts.pkl`: Contains expert data.
- `teams_sorted.pkl`: Contains sorted team data.
- `teamsvecs.pkl`: Vector representations of teams.

If the `experts.pkl` or `teams_sorted.pkl` files are not found, they will be generated automatically using the provided `teamsvecs.pkl` file.

## Results
<p align="center">
   <img src="https://github.com/user-attachments/assets/95980f62-df40-414b-bdda-a54bd7f401fa" alt="Image 1" width="150" />
   <img src="https://github.com/user-attachments/assets/503f1788-35e7-45b4-887d-fafd665869da" alt="Image 2" width="800"/></p>
<br>

<p align="center">
   <img src="https://github.com/user-attachments/assets/74252e18-2a97-4390-8783-5870ce6438e3" alt="Image 1" width="150" />
   <img src="https://github.com/user-attachments/assets/42b0177c-c739-4184-9bb1-7ac0b6a8eaee" alt="Image 1" width="400" />
</p>

<p align="center">
   <img src="https://github.com/user-attachments/assets/2c0faf6d-8697-4e86-973e-a69a3e61a02c" alt="Image 1" width="150" />
   <img src="https://github.com/user-attachments/assets/fa0bf77a-44cd-4881-b5e6-21c793fcc4d8" alt="Image 1" width="800" />
</p>

## License
[MIT LICENSE](https://github.com/fani-lab/OpenGNTF/blob/main/LICENSE)

## Acknowledgments

- [PyG](https://pyg.org/)
- [PyTorch](https://pytorch.org/)
- Graph Neural Network models like GIN, GAT, and more.

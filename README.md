# GNN Subgraph Training and Evaluation

OpenGNTG focuses on training and evaluating Graph Neural Networks (GNNs) using subgraphs of different types (e.g., Skill/Team/Expert (STE), Skill/Expert (SE), and STEL). The pipeline processes datasets, prepares subgraph data, and trains specified GNN models with user-defined parameters.

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

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License
[MIT LICENSE](https://github.com/fani-lab/OpenGNTF/blob/main/LICENSE)

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- Graph Neural Network models like GIN, GAT, and more.

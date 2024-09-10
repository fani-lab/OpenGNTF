source ../venvs/opentf_tmp/bin/activate;
cd ../src;

# Define the parameter values
gnn_models=("gs" "gin" "gat" "gatv2")
graph_types=("SE" "STE" "STEL")
full_subgraphs=(0)
eval_methods=("sum" "fusion")
data_path="../data/dblp/"
dims=(32 64 128)
num_neighbors=("20" "20 10" "20 10 5")

# Iterate through all combinations of the parameters
for gnn_model in "${gnn_models[@]}"; do
    for graph_type in "${graph_types[@]}"; do
    	for d in "${dims[@]}"; do
    		for nn in "${num_neighbors[@]}"; do
		        for full_subgraph in "${full_subgraphs[@]}"; do
		            for eval_method in "${eval_methods[@]}"; do
		                echo "Running: gnn_model=${gnn_model}, graph_type=${graph_type}, num_neighbors = ${nn}, dim = ${d}, full_subgraph=${full_subgraph}, eval_method=${eval_method}"
		                python -u main.py --data_path ${data_path} --gnn_model ${gnn_model} --graph_type ${graph_type} --num_neighbors ${num_neighbors} --dim ${d} --full_subgraph ${full_subgraph} --eval_method ${eval_method}
		                echo "Complete: gnn_model=${gnn_model}, graph_type=${graph_type}, num_neighbors = ${nn}, dim = ${d}, full_subgraph=${full_subgraph}, eval_method=${eval_method}"
		            done
		        done
		    done
		done
    done
done

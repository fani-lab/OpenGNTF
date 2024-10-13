import os
import pandas as pd

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Combine CSV files and add settings column.')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing CSV files')
parser.add_argument('--output_file', type=str, default='combined_results.csv', help='Name of the output CSV file')
args = parser.parse_args()

# Assign folder path and output file from arguments
folder_path = args.folder_path
output_file = args.output_file

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty list to hold dataframes
dfs = []

# Loop through each CSV file
for csv_file in csv_files:
    print(f'Processing file : {csv_file}')
    # Extract the settings from the file name (remove .csv)
    setting_name = csv_file.replace('.csv', '')
    
    # Read the CSV file into a dataframe
    df = pd.read_csv(os.path.join(folder_path, csv_file))
    
    print(f'----df : {df}')

    # Add a new column 'settings' with the setting name
    df.insert(0, 'settings', setting_name)
    
    # Append the dataframe to the list
    dfs.append(df)

# Concatenate all dataframes in the list into one dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV created: {output_file}")


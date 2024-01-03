import pandas as pd

# Input CSV file path
input_file_path = './modified_file.csv'  # Replace with the path to your input CSV file

# Output CSV file path
output_file_path = './output_file.csv'  # Replace with the desired path for the output CSV file

# Number of lines to select from each chunk
lines_per_chunk = 1000

# Initialize an empty DataFrame to store the selected lines
selected_lines_df = pd.DataFrame()

# Read the CSV file in chunks
chunk_size = 1000000
for chunk in pd.read_csv(input_file_path, chunksize=chunk_size):
    # Select the specified number of lines from each chunk where the label equals 0
    selected_chunk = chunk.head(2 * lines_per_chunk) if (chunk['Label'] == 0).any() else chunk.head(lines_per_chunk)

    # Append the selected lines to the result DataFrame
    selected_lines_df = pd.concat([selected_lines_df, selected_chunk], ignore_index=True)

# Save the result to a new CSV file with the same header
selected_lines_df.to_csv(output_file_path, index=False)

print(f'Selected lines saved to: {output_file_path}')


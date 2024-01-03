import pandas as pd

# Read the CSV file
file_path = '/home/elbouazz/Desktop/PhD/PhD Report mohamed EL BOUAZZATI/PhD-report/Chapitre2/figures/results/dataset_simulation_dummymac.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Modify the label column
df['Label'] = df['Label'].replace({'stack_overflow(S1)': 1, 'heap_overflow(S2)': 1, 'legitimate(S0)': 0})
# colors = {'legitimate(S0)': 'g', 'heap_overflow(S2)': 'r', 'stack_overflow(S1)': 'b'}  # Correct colors dictionary
# Save the modified DataFrame to a new CSV file
output_file_path = 'modified_file.csv'  # Replace with the desired output path
df.to_csv(output_file_path, index=False)

print(f'Modified CSV saved to: {output_file_path}')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the data from a CSV file
data = pd.read_csv('dataset_simulation_dummymac.csv')

# Drop the 'PIP_STALL' and 'JMP_STALL' columns
data = data.drop(columns=['PIP_STALL', 'JMP_STALL'])

# Create a figure with 10 subplots arranged in a 2x5 grid
fig, axs = plt.subplots(2, 5, figsize=(20, 13))
plt.subplots_adjust(bottom=0.15) 
colors = {'legitimate(S0)': 'g', 'heap_overflow(S2)': 'r', 'stack_overflow(S1)': 'b'}  # Correct colors dictionary
textsize=20
# Define a dictionary with bin counts for each feature
bins_dict = {
    'Cycles': 150,
    'Minstret': 70,
    'LD_STALL': 70,
    'IMISS': 150,
    'LD': 50,
    'ST': 50,
    'JUMP': 50,
    'BRANCH': 70,
    'BRANCH_TAKEN': 70,
    'COMP_INSTR': 80
}  # Customize this

# Loop through each feature
for i, feature in enumerate(data.columns[:-1]):
    # Get the x and y coordinates for the subplot
    x = i // 5
    y = i % 5
    ax = axs[x, y]
    
    # Get bin count for the current feature
    bins = bins_dict.get(feature, 30)  # Default to 30 if feature not in dictionary

    # Create a histogram for the feature and target
    for t in data['Label'].unique():
        ax.hist(data[data['Label'] == t][feature], bins=bins, alpha=0.9, label=f'{t}', color=colors[t])
    ax.set_title(feature)
    ax.xaxis.label.set_size(textsize)
    ax.yaxis.label.set_size(textsize)
    ax.title.set_size(textsize)
    ax.set_xlabel('HPC Value')
    ax.set_ylabel('Frequency')
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    ax.tick_params(axis='both', which='major', labelsize=0.8*textsize)  # Set tick labels fontsize

fig.subplots_adjust(bottom=0.12, hspace=0.8, wspace=0.8)
# Add a legend for the whole figure
handles, labels = axs[0, 0].get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.019), ncol=3, fontsize=textsize)

# Add a title for the figure
# fig.suptitle("Histograms for each feature and target")


plt.tight_layout()
plt.savefig("./../packet_injection_dataset.pdf", format='pdf',dpi=500, bbox_inches='tight')


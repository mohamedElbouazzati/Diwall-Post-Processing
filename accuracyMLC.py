import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist

path = './classifier_comparision.csv'

df = pd.read_csv(path)
df1 = pd.DataFrame(df)

# set width of bar
barWidth = 0.18
fig = plt.subplots(figsize=(12, 8))

# set height of bar
IT = np.array(df1.Accuracy)
ECE = np.array(df1.Precision)
CSE = np.array(df1.Recall)
F1 = np.array(df1['F1-Score'])

# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, IT, color='c', width=barWidth,
        edgecolor='grey', label='Accuracy', alpha=0.75)
plt.bar(br2, ECE, color='g', width=barWidth,
        edgecolor='grey', label='Precision', alpha=0.75)
plt.bar(br3, CSE, color='b', width=barWidth,
        edgecolor='grey', label='Recall', alpha=0.5)
plt.bar(br4, F1, color='y', width=barWidth,
        edgecolor='grey', label='F1-Score', alpha=0.75)

# Adding Xticks
plt.ylabel('Metrics :(%)', fontweight='bold', fontsize=15)
plt.yticks(fontsize=15.0, fontweight='bold')
plt.xticks([r + barWidth for r in range(len(IT))], df.Classifier,
           rotation=90, fontsize=17.0, fontweight='bold')
plt.title('Metrics Comparison of Classifiers Models',
          fontsize=15.0, fontweight='bold')
plt.legend(fontsize=15.0, loc='upper center', bbox_to_anchor=(0.5, 0.1),
           fancybox=True, shadow=True, ncol=4)

# Add a line at 90%
plt.axhline(y=90, color='red', linestyle='--', linewidth=1)
plt.text(len(IT) - 0.5, 90, '90%', ha='right', va='bottom',
         color='red', fontsize=12, fontweight='bold')

# Customize grid appearance
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.8)

# Adjust layout for better visualization
plt.tight_layout()

# Save the plot as a PDF file
plt.savefig('accuracy_multi_42_with_line.pdf')

# Show the plot (optional)
plt.show()

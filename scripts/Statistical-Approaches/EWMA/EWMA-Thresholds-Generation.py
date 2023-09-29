import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def exponential_weighted_mean(data, alpha=0.5):
    ewma = [data[0]]
    for i in range(1, len(data)):
        ewma.append(alpha * data[i] + (1 - alpha) * ewma[i-1])
    return ewma

def calculate_UCL_LCL(ewma, sigma=3):
    mean = np.mean(ewma)
    std = np.std(ewma)
    UCL = mean + sigma * std
    LCL = mean - sigma * std
    return UCL, LCL

# Load CSV file
df = pd.read_csv("ewma_data.csv")

# Calculate EWMA
ewma = exponential_weighted_mean(df['RSSI'].values)

# Calculate UCL and LCL
UCL, LCL = calculate_UCL_LCL(ewma)

# Add EWMA, UCL, and LCL columns to dataframe
df['ewma'] = ewma
df['UCL'] = UCL
df['LCL'] = LCL

# Plot data, EWMA, UCL, and LCL
plt.plot(df.index, df['ewmaRSSI'], label='Data')
plt.plot(df.index, df['ewma'], label='EWMA')
plt.plot(df.index, [df['UCL']] * len(df), label='UCL')
plt.plot(df.index, [df['LCL']] * len(df), label='LCL')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Data, EWMA, UCL, and LCL Plot')
plt.show()
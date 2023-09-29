import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('Dataset_farTx_nearjamtrig.csv')
df1 = pd.read_csv('Dataset_farTx_nearjamcont.csv')

fig = plt.figure(figsize=(10,6))
plt.subplots_adjust(bottom=0.15) 
# Create a scatter plot for SNR vs Packet Size
ax1 = fig.add_subplot(223)
line1 = ax1.plot(df.index, df['SNR'], label='SNR', color='blue')
ax1.set_title('SNR - Trigger mode')  # Set the title
ax1.set_xlabel('Packet index')
ax1.set_ylabel('SNR')
ax1.grid()
ax1.plot([3300, 3300], [df['SNR'].min(), df['SNR'].max()],ls=':', lw=1.5,color='r')
ax1.plot([3800, 3800], [df['SNR'].min(), df['SNR'].max()],ls=':', lw=1.5,color='r')
ax1.annotate("Jam Part",xy=(3000, df['SNR'].min()),xytext=(4000,df['SNR'].min()+1), fontsize=12,color='r')

# Create a scatter plot for RSSI vs Packet Size
ax2 = fig.add_subplot(221)
line2 = ax2.plot(df.index, df['RSSI'], label='RSSI', color='green')
ax2.set_title('RSSI - Trigger mode')  # Set the title
ax2.set_xlabel('Packet index')
ax2.set_ylabel('RSSI')
ax2.grid()
ax2.plot([3300, 3300], [df['RSSI'].min(), df['RSSI'].max()],ls=':', lw=1.5,color='r')
ax2.plot([3800, 3800], [df['RSSI'].min(), df['RSSI'].max()],ls=':', lw=1.5,color='r')
ax2.annotate("Jam Part",xy=(3000, df['RSSI'].min()),xytext=(4000,df['RSSI'].min()+5), fontsize=12,color='r')

ax3 = fig.add_subplot(224)
ax3.plot(df1.index, df1['SNR'], label='SNR', color='blue')
ax3.set_title('SNR - Continuous mode')  # Set the title
ax3.set_xlabel('Packet index')
ax3.set_ylabel('SNR')
ax3.grid()
ax3.plot([2990, 2990], [df1['SNR'].min(), df1['SNR'].max()],ls=':', lw=1.5,color='r')
ax3.plot([4020, 4020], [df1['SNR'].min(), df1['SNR'].max()],ls=':', lw=1.5,color='r')
ax3.annotate("JamPart",xy=(3100, df1['SNR'].min()),xytext=(3050,df1['SNR'].min()+1), fontsize=12,color='r')

ax4 = fig.add_subplot(222)
ax4.plot(df1.index, df1['RSSI'], label='RSSI', color='green')
ax4.set_title('RSSI - Continuous mode')  # Set the title
ax4.set_xlabel('Packet index')
ax4.set_ylabel('RSSI')
ax4.grid()
ax4.plot([2990, 2990], [df1['RSSI'].min(), df1['RSSI'].max()],ls=':', lw=1.5,color='r')
ax4.plot([4020, 4020], [df1['RSSI'].min(), df1['RSSI'].max()],ls=':', lw=1.5,color='r')
ax4.annotate("Jam Part",xy=(3100, df1['RSSI'].min()),xytext=(3050,df1['RSSI'].min()+5), fontsize=12,color='r')
fig.subplots_adjust(bottom=0.12, hspace=0.8, wspace=0.8)
fig.legend([line1[0], line2[0]], ['SNR', 'RSSI'],loc='lower center', bbox_to_anchor=(0.5, -0.019),fancybox=True, shadow=True, ncol=2,fontsize=14.0)
fig.tight_layout()

plt.savefig("../dataset_jamming.pdf", dpi=300,bbox_inches='tight')

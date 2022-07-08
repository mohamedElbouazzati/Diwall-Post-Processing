import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import pyplot
import matplotlib.pyplot as plt
path1='/home/med/Desktop/workflow/handsonML/legitimate_Dataset.csv'
df1=pd.read_csv(path1)
df2=pd.read_csv('/home/med/Desktop/workflow/handsonML/label/1000_same_size_35_42/Packet_Dataset.csv')
path='/home/med/Desktop/workflow/handsonML/Buffer_Dataset.csv'
df=pd.read_csv(path)
#df.plot(x ='Cycle', y='HPM2(INSTR)', color='r')
#sns.lineplot(x ='Cycle', y='Packet_Nbre',data=df2)
#df2["Cycle"].plot.hist(bins=80)
x1 = df2.loc[df2.Packet_Nbre=='legitime', 'Minstret']
x2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'Minstret']
x3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'Minstret']
y1 = df2.loc[df2.Packet_Nbre=='legitime', 'COMP_INSTR']
y2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'COMP_INSTR']
y3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'COMP_INSTR']
z1 = df2.loc[df2.Packet_Nbre=='legitime', 'LD']
z2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'LD']
z3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'LD']
r1 = df2.loc[df2.Packet_Nbre=='legitime', 'ST']
r2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'ST']
r3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'ST']
t1 = df2.loc[df2.Packet_Nbre=='legitime', 'Cycles']
t2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'Cycles']
t3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'Cycles']
g1 = df2.loc[df2.Packet_Nbre=='legitime', 'BRANCH_TAKEN']
g2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'BRANCH_TAKEN']
g3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'BRANCH_TAKEN']
k1 = df2.loc[df2.Packet_Nbre=='legitime', 'PIP_STALL']
k2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'PIP_STALL']
k3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'PIP_STALL']
f1 = df2.loc[df2.Packet_Nbre=='legitime', 'BRANCH']
f2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'BRANCH']
f3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'BRANCH']
d1 = df2.loc[df2.Packet_Nbre=='legitime', 'JMP_STALL']
d2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'JMP_STALL']
d3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'JMP_STALL']
p1 = df2.loc[df2.Packet_Nbre=='legitime', 'JUMP']
p2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'JUMP']
p3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'JUMP']
u1 = df2.loc[df2.Packet_Nbre=='legitime', 'IMISS']
u2 = df2.loc[df2.Packet_Nbre=='stack_overflow', 'IMISS']
u3 = df2.loc[df2.Packet_Nbre=='heap_overflow', 'IMISS']

kwargs = dict(alpha=0.5, bins=100)

red_patch = mpatches.Patch(color='r', label='Stack Overflow')
red_patch2 = mpatches.Patch(color='b', label='Heap Overflow')
red_patch1 = mpatches.Patch(color='g', label='Benign Packet')
#JMP_STALL,IMISS,LD,ST,JUMP,BRANCH,BRANCH_TAKEN,COMP_INSTR,PIP_STALL
plt.suptitle(fontsize=24.0 ,fontweight='bold', t='Frequency Histogram of Hardware Events')
plt.figlegend(fontsize=24.0, handles=[red_patch1,red_patch,red_patch2])


plt.subplot(5, 2, 1)

plt.hist(x1, **kwargs, color='g')
plt.hist(x2, **kwargs, color='r')  
plt.hist(x3, **kwargs, color='b')  
plt.gca().set_title('Minstret',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')

plt.subplot(5, 2, 2)
plt.hist(z1, **kwargs, color='g')
plt.hist(z2, **kwargs, color='r')
plt.hist(z3, **kwargs, color='b')  
plt.gca().set_title('LD',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')

plt.subplot(5, 2, 3)
plt.hist(u1, **kwargs, color='g')
plt.hist(u2, **kwargs, color='r')
plt.hist(u3, **kwargs, color='b')  
plt.gca().set_title('IMISS',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')
plt.subplot(5, 2, 4)
plt.hist(r1, **kwargs, color='g')
plt.hist(r2, **kwargs, color='r')
plt.hist(r3, **kwargs, color='b')  
plt.gca().set_title('ST',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')

plt.subplot(5, 2, 5)
plt.hist(t1, **kwargs, color='g')
plt.hist(t2, **kwargs, color='r')
plt.hist(t3, **kwargs, color='b')  
plt.gca().set_title('Cycles',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')

plt.subplot(5, 2, 6)
plt.hist(g1, **kwargs, color='g')
plt.hist(g2, **kwargs, color='r')
plt.hist(g3, **kwargs, color='b')  
plt.gca().set_title('BRANCH_TAKEN',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')

plt.subplot(5, 2, 7)
plt.hist(p1, **kwargs, color='g')
plt.hist(p2, **kwargs, color='r')
plt.hist(p3, **kwargs, color='b')  
plt.gca().set_title('JMP',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')

plt.subplot(5, 2, 8)
plt.hist(k1, **kwargs, color='g')
plt.hist(k2, **kwargs, color='r')
plt.hist(k3, **kwargs, color='b')  
plt.gca().set_title('PIP_STALL',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')
plt.subplot(5, 2, 9)
plt.hist(d1, **kwargs, color='g')
plt.hist(d2, **kwargs, color='r')
plt.hist(d3, **kwargs, color='b')  
plt.gca().set_title('JMP_STALL',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')
plt.subplot(5, 2, 10)
plt.hist(y1, **kwargs, color='g')
plt.hist(y2, **kwargs, color='r')
plt.hist(y3, **kwargs, color='b')  
plt.gca().set_title('COMP_INSTR',fontsize=17.0,fontweight='bold')
plt.gca().set_ylabel('Frequency',fontsize=17.0,fontweight='bold') 
plt.xticks(fontsize=17.0,fontweight='bold')
plt.yticks(fontsize=17.0,fontweight='bold')
#df["Cycle"].plot(kind='hist')




plt.show()
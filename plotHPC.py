
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
path='/home/med/Desktop/workflow/handsonML/Packet_Dataset.csv'
df=pd.read_csv(path)
l=['Cycles','Minstret','JMP_STALL','IMISS','LD','ST','JUMP','BRANCH','BRANCH_TAKEN','COMP_INSTR','PIP_STALL']

df.plot(y ='Packet_Nbre', x='Minstret',kind = 'scatter')
plt.show()
#JMP_STALL,IMISS,LD,ST,JUMP
# Cycles,Minstret,JMP_STALL,IMISS,LD,ST,JUMP,BRANCH,BRANCH_TAKEN,COMP_INSTR,PIP_STALL
from tkinter import Pack
import dask.dataframe as dd
import csv
import pandas as pd
import numpy as np

dtype1={'Packet_Nbre': 'float64',
      'Cycle' :'float64',
      'HPM0(Cycles)':'float64',
      'HPM2(INSTR)':'float64',
      'HPM3(LD_STALL)':'float64',
      'HPM4(JMP_STALL)':'float64',
      'HPM5(IMISS)':'float64',
      'HPM6(LD)':'float64',
      'HPM7(ST)':'float64',
      'HPM8(JUMP)':'float64',
      'HPM9(BRANCH)':'float64',
      'HPM10(BRANCH_TAKEN)': 'float64',
       'HPM11(COMP_INSTR)': 'float64',
       'HPM7(ST)': 'object',
       'HPM8(JUMP)': 'float64',
       'HPM9(BRANCH)': 'float64'}

header="Packet_Nbre,Cycle,HPM0,HPM2,HPM3,HPM4,HPM5,HPM6,HPM7,HPM8,HPM9,HPM10,HPM11\n"
csv_file = open("Packet_Dataset.csv", "w")
csv_file.write(header)       
dask_df = dd.read_csv('HPMtracer.csv',dtype=dtype1)

vector_iter=np.array(dask_df.Packet_Nbre)
Packet_number = int(np.max(vector_iter))
i=0
for i in range(Packet_number+1):
 df1=dask_df[dask_df.Packet_Nbre==i] 
 datapacket=pd.DataFrame(df1).astype(np.int64)
 packeti=datapacket.iloc[-1]-datapacket.iloc[0]
 packet=pd.DataFrame(packeti)
 packet=packet.transpose()
 if i<int(Packet_number/2):
    packet[0]="legitime"
 else : packet[0]="Buf_over"
 csv_file = open("Packet_Dataset.csv", "a")
 packet.to_csv('Packet_Dataset.csv',sep=',',header=False, index=False,encoding='utf-8',mode="a") 
csv_file.close() 

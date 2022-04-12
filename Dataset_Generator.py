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
 packet[0]=i
 packet.to_csv('Packet_Dataset.csv',sep=',',header=False, index=False,encoding='utf-8',mode="a")
csv_file.close() 
















'''

i=1
packeti['Packet_Nbre']=packeti['Packet_Nbre'].replace(0.0,i)
dict[1]=packeti
print(packeti)

for i in range(int(Packet_number)):
 #arr = np.where(vector_iter == i) # index of  packet n 
 #arr2 = np.where(vector_iter == i+1) #  index packet n + 1
 df1=dask_df[dask_df.Packet_Nbre==i] 
 df2=dask_df[dask_df.Packet_Nbre==i+1] 
 datapacket=df1.compute()
 #datapacket=pd.DataFrame(df1) # faster 
 print(datapacket.iloc[-1:]-datapacket.iloc[0:])
 exit()
 
 x=arr[0][-1]
 print(x)
 print(arr[0][-1])
 #print(dask_df.loc[1,-1])
 print(dask_df.loc[4:10.5])
 
 l=[]
 l=dask_df.iloc[0]
 print(l[-1])
 print(dask_df.loc[arr[0][-1],:])
 
 last = dask_df.loc[arr[0][-1]] #  data of packet n
 first =dask_df.loc[arr2[0][-1]]  # data of packet n + 1

 arrt=first-last  # difference  data packet 
    #string ='conparison of packet '+ str(i)+ 'with packet ' + str(i+1) 
            #string="packet:"+str(i+1)
 #arrt[0]=i+1
 #dict[i]=arrt

 print("\n")
 print(arrt)
 exit()
    #print(dict)

# print(data.Packet_Nbre)

# sprint(maxi)
# dictionnaire

exit()
df = dd.DataFrame(dict)
df=df.transpose()    
df.to_csv('testo.csv',sep=',',header=True, index=False,encoding='utf-8')
'''
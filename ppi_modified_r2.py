# To remove nodes which doesnt have lfq intensity values


# Creating ppi network

import config as cnf
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random as rnd


# Removing


label = {'name':['Apoe','Egfr','Clu','Grn','Vtn','Lrp1','Gsn','Reln','Mup12', 'Mup19', 'Mug1', 'Lifr', 'Itih1', 'Hgfac', 'Ubtfl1', 'Orm2', 'Spp2', 'Amy2a2'],'value':[0.3125,0.3125,0.5,0.375,0.1875,0.5625,0.1875,0.1875,0,0,0,0,0,0,0,0,0,0]}

label_data = pd.DataFrame(data=label)
df =pd.read_csv('string_interactions_round_2.tsv',sep='\t')








feat = pd.read_csv(cnf.datapath+'tbi_r2_t5_norm.csv')

all_proteins = feat['name'].values.tolist()



df_updated = pd.DataFrame()
# df_updated = df.copy(deep=True)
list_indices = []
for index, row in df.iterrows():
    if ((df["node1"][index] in all_proteins) and (df["node2"][index] in all_proteins)):
        list_indices.append(index)
        # df_updated.append(df.iloc[index])
        # df_updated.drop(df_updated.index[index])

df_updated = df.iloc[list_indices]
# df_updated = df.drop(df.index[np.where(df["node1"] not in all_proteins or df["node2"] not in all_proteins)])


###### Unique mapping ####
list1 = df_updated["node1"].values.tolist()
list2 = df_updated["node2"].values.tolist()
list3 = list1+list2
list4 = np.unique(list3)
# list4 = np.unique(list3).tolist()


id_list =np.arange(196).tolist()

mapping_proteins = pd.DataFrame({'name':list4,'id':id_list})
df_mapped = df_updated.copy()
for index, row in df_mapped.iterrows():
    row['node2']= mapping_proteins.iloc[np.where(row['node2']==mapping_proteins.name)[0][0],1]
    row['node1'] = mapping_proteins.iloc[np.where(row['node1'] == mapping_proteins.name)[0][0], 1]
    df_mapped.loc[index,'node1'] =row['node1']
    df_mapped.loc[index,'node2'] =row['node2']



################################## Making graph with mapped feature ##########################

G = nx.from_pandas_edgelist(df_mapped,'node1','node2', edge_attr='combined_score')


# i = 0
for v in G.nodes():
    u = mapping_proteins.iloc[np.where(v==mapping_proteins.id)[0],0].to_numpy()[0]
    G.nodes[v]['feature'] =  feat.iloc[np.where(u==feat.name)[0],1:12].to_numpy()

    if np.any(np.where(u == label_data.name)[0]):
        G.nodes[v]['label']= label_data.iloc[np.where(u == label_data.name)[0], 1].to_numpy()
    else:
        G.nodes[v]['label']= -1




for v1,v2 in G.edges:

   G.edges[v1,v2]['feature'] = df_mapped.iloc[np.where(np.logical_or(np.logical_and (df_mapped.node1 == v1, df_mapped.node2 ==v2),np.logical_and (df_mapped.node1 == v2, df_mapped.node2 ==v1)))[0][0]]['combined_score']

############################################################################################
# print('check')
# Saving the graph to the data


filepath  = cnf.modelpath+'TBI_t5.pkl'
with open(filepath, 'wb') as f:
        pickle.dump(G, f)


filepath  = cnf.modelpath+'modified_proteins'
with open(filepath, 'wb') as f:
        pickle.dump(mapping_proteins, f)

nx.draw_networkx(G,with_labels=True)
plt.show()

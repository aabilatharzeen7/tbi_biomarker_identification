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

# list_1 = ['C4bp','Hc','Fcna','Car2','Trf','C1s1','C1s2','Hbb-bs','Pzp','Ric8','Amy2a2','Mup19','Mup18','Mup12','Hba-a2']

label = {'name':['Apoe','Egfr','Clu','Grn','Vtn','Lrp1','Gsn','Reln','Mup12', 'Mup19', 'Mug1', 'Lifr', 'Itih1', 'Hgfac', 'Ubtfl1', 'Orm2', 'Spp2', 'Amy2a2'],'value':[0.3125,0.3125,0.5,0.375,0.1875,0.5625,0.1875,0.1875,0,0,0,0,0,0,0,0,0,0]}

label_data = pd.DataFrame(data=label)
df =pd.read_csv('string_interactions_round1.tsv',sep='\t')

df3 = df.sample( frac= 0.02, replace=False, weights=None, random_state=None, axis=None)


# df2 = df[~df.isin(list_1).node1 |  ~df.isin(list_1).node2]



feat = pd.read_csv('tbi_r1_dataFeat_backup2.csv')

G = nx.from_pandas_edgelist(df3,'node1','node2', edge_attr='combined_score')

# i = 0
for u in G.nodes():
    G.nodes[u]['feature'] =  feat.iloc[np.where(u==feat.name)[0],1:30].to_numpy()
    # G.nodes[u]['node_id'] = i
    # i = i+1
    if np.any(np.where(u == label_data.name)[0]):
        G.nodes[u]['label']= label_data.iloc[np.where(u == label_data.name)[0], 1].to_numpy()


    # G.nodes[u]['label'] = rnd.randint(0, 1)

for v1,v2 in G.edges:
    G.edges[v1,v2]['feature'] = df.iloc[np.where(np.logical_or(np.logical_and (df.node1 == v1, df.node2 ==v2),np.logical_and (df.node1 == v2, df.node2 ==v1)))[0][0]]['combined_score']



# Saving the graph to the data


filepath  = cnf.modelpath+'TBI_1.pkl'
with open(filepath, 'wb') as f:
        pickle.dump(G, f)


nx.draw_networkx(G,with_labels=True)
plt.show()
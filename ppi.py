# Creating ppi network

import config as cnf
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random as rnd


df =pd.read_csv('string_interactions_short.tsv',sep='\t')
G = nx.from_pandas_edgelist(df,'node1','node2', edge_attr='combined_score')


for u in G.nodes():
    G.nodes[u]['feature'] =  np.random.rand(1,10)
    G.nodes[u]['label'] = rnd.randint(0, 1)

    # print(node_features[0] / speed[counttime][u])  # for getting the normalization rescale factor

# Saving the graph to the data



filepath  = cnf.modelpath+'TBI.pkl'
with open(filepath, 'wb') as f:
        pickle.dump(G, f)


nx.draw_networkx(G,with_labels=True)
plt.show()
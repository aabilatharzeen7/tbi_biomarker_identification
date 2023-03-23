# Creating ppi network

import config as cnf
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random as rnd


<<<<<<< Updated upstream
df =pd.read_csv('string_interactions_short.tsv',sep='\t')
=======
df =pd.read_csv('string_interactions_short_final.tsv',sep='\t')
>>>>>>> Stashed changes
G = nx.from_pandas_edgelist(df,'node1','node2', edge_attr='combined_score')


for u in G.nodes():
    G.nodes[u]['feature'] =  np.random.rand(1,10)
    G.nodes[u]['label'] = rnd.randint(0, 1)

    # print(node_features[0] / speed[counttime][u])  # for getting the normalization rescale factor

# Saving the graph to the data

<<<<<<< Updated upstream
=======
var = G.nodes
import csv

# opening the csv file in 'w+' mode


# writing the data into the file



with open('tbi_round1_v3.csv', 'w') as ofile:
        outfile = csv.writer(ofile)
        outfile.writerows(([str(i)] for i in var))


df = pd.read_csv("tbi_round1_v3.csv")


>>>>>>> Stashed changes


filepath  = cnf.modelpath+'TBI.pkl'
with open(filepath, 'wb') as f:
        pickle.dump(G, f)


nx.draw_networkx(G,with_labels=True)
plt.show()
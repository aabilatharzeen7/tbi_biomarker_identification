# Creating ppi network


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



df =pd.read_csv('string_interactions_short.tsv',sep='\t')
G = nx.from_pandas_edgelist(df,'node1','node2', edge_attr='combined_score')
nx.draw(G, with_labels = True)
plt.show()
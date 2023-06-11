# Plot the PPI graph for demonstration purpose


import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Load string interaction database
df_1 =pd.read_csv('string_interactions_round1.tsv',sep='\t')


# nodes chosen for drawing graph
list_ = ['Apoe','Clu','Lrp1','Reln','Lifr',  'Orm2', 'Hgfac','Hpx','Apoc1','Pltp','Hspa5','Serpinf2','Pzp','Thbs1','Cpn2','Fn1','Gapdh','Igfals','Azgp1','C1qa','C1s1','C3','F12','Bche','Col1a1','Efemp1','F13b','Mup18','Plg','Gpx3']

# Nodes chosen with confidence score
df = {'node':['Apoe','Clu','Lrp1','Reln','Lifr',  'Orm2', 'Hgfac','Hpx','Apoc1','Pltp','Hspa5','Serpinf2','Pzp','Thbs1','Cpn2','Fn1','Gapdh','Igfals','Azgp1','C1qa','C1s1','C3','F12','Bche','Col1a1','Efemp1','F13b','Mup18','Plg','Gpx3'],
'color':[0.3125,0.5,0.5625,0.1875,0,0,0,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5]}

color_map_nodes = ['salmon','orangered','red','deepskyblue','blue','blue','blue','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow']

# df = {'node':['Apoe','Clu','Lrp1','Reln','Lifr',  'Orm2', 'Hgfac','Hpx','Apoc1','Pltp','Hspa5','Serpinf2','Pzp','Thbs1','Cpn2','Fn1','Gapdh','Igfals','Azgp1','C1qa','C1s1','C3','F12','Bche','Col1a1','Efemp1','F13b','Mup18','Plg','Gpx3'],
# 'color':[0.3125,0.5,0.5625,0.1875,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
df = pd.DataFrame(data=df)
df = df.set_index('node')


# Filtering the string interaction database dataframe based on nodes chosen

df2 = df_1[(df_1.isin(list_).node1 & df_1.isin(list_).node2)]
# df2 = df_1[(df_1.isin(list_).node1 & df_1.isin(list_).node2) & (df_1.combined_score > 0.4)] # cannot use combine score filter, to use this modify the confidence score dictionary/dataframe as well

# Create graph
G = nx.from_pandas_edgelist(df2,'node1','node2', edge_attr='combined_score')

for v1,v2 in G.edges:
    G.edges[v1,v2]['feature'] = df_1.iloc[np.where(np.logical_or(np.logical_and (df_1.node1 == v1, df_1.node2 ==v2),np.logical_and (df_1.node1 == v2, df_1.node2 ==v1)))[0][0]]['combined_score']

# making dictionary of edge weights

edge_labels = dict([((n1, n2), d['feature'])
                    for n1, n2, d in G.edges(data=True)])


# Set position and color bar settings

layout = nx.spring_layout(G)
vmin = df['color'].min()
vmax = df['color'].max()
cmap = plt.cm.coolwarm

edges, weights = zip(*nx.get_edge_attributes(G, 'feature').items())
# Draw complete graph and the second draw just for edge with labels
nx.draw_networkx(G, pos=layout,  with_labels=True, node_color=color_map_nodes , font_color='black',font_weight = 'bold',width =0.5,font_size=6,node_size = 500)



############################## Edge Plots #####################################################
## Grey edge and label
# nx.draw_networkx(G, pos=layout,  with_labels=True, node_color=df['color'],cmap=cmap, vmin=vmin, vmax=vmax, font_color='white',edge_color= 'grey',width =0.5,font_size=6,node_size = 500)


## Edge label
# nx.draw_networkx_edge_labels(G, pos=layout,edge_color=edge_labels ,font_color='red',font_size=3)


## Line weight
# nx.draw_networkx_edges(G = G, pos = layout, edge_color='k', alpha=0.6, width=weights)

## Edge color

options = {
    "edge_color": weights,
    "width":0.6,
    "edge_cmap": plt.cm.Dark2,
}
# nx.draw_networkx_edges(G,  pos = layout, edge_color=weights,width =0.5)
nx.draw_networkx_edges(G, pos = layout, **options)

# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# sm.set_array([])
# cbar = plt.colorbar(sm,shrink=0.5)


# plt.show()



# plt.show(block=False)

# plt.savefig('PPI_1.png', format= 'png',dpi=1200)
plt.show()
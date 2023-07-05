# Plot the PPI graph with the results obtained in eacg round


import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Load string interaction database
df_1 =pd.read_csv('string_interactions_short_combined.tsv',sep='\t')


# nodes chosen for drawing graph
list_1 = ['Orm1','Apoa1','Apob','Hspa1b','Amy1', 'Mup17', 'Thbs1','Psma4','Prdx2','Pltp','Hspa5','Lcat','Vcp','Thbs1','Cpn2','Aldoa','Gapdh','Psmb5','Azgp1','C1qa','Rpl7','C3','Cfd','Mmrn1','Apoc1','Idh2','F13b','A2mp','Kng1','Gpx3']
list_2 = ['Aldob','Apoc3','Thbs1','Psma5','Orm2', 'Apoa2', 'Orm1','Apoa1','F7','C1qa','Hspa5','Lcat','Vcp','Pros1','Cpn2','Fgb','Gapdh','Col1a1','Azgp1','Mst1','Rpl7','C3','Cfd','Mmrn1','Clu','Idh2','F13b','A2mp','Kng1','Gpx3']
list_3 = ['Orm1','Cir1','Thbs4','Apoa2','Serpina1c', 'Acta2', 'Cat','Psma2','F7','C1qa','Hspa5','Lcat','Vcp','Pros1','Cpn2','Fgb','Gapdh','Col1a1','Lifr','Pkm','Hp','C3','Spp2','Cpb2','Hspa5','Idh2','F13b','A2mp','Kng1','Gpx3']
list_4 = ['Thbs1','Orm1','Apoa2','Apoc2','Amy1', 'Psma5', 'Aldob','Psmb6','Serpina1e','C1qa','Hspa5','Lcat','Vcp','Pros1','Saa1','Fgb','Apcs','Cfd','Fbln1','Pkm','Hp','B2m','Spp2','Cpb2','Hspa5','Idh2','F13b','A2mp','Kng1','Gpx3']
list_5 = ['Orm2','Psma4','Apoa1','Saa2','Psmb8', 'Lcat', 'Thbs4','Serpina1b','Aldh1a7','Thbs1','Hspa5','Ttr','Vcp','Pros1','Saa1','Fgb','Apcs','Cfd','Fbln1','Pkm','Hp','B2m','Itih3','Cpb2','Pkm','Idh2','F13b','C5','Alb','Gpx3']
list_ = ['Thbs4','Hspa1a','Apoa2','Afm','Egfr', 'Orm1', 'Apob','Apoa1','Orm2','Apcs','Cd5l','Ttr','Hspa8','Pros1','Saa1','Fgb','F2','Cfd','Fbln1','Pkm','Igfals','B2m','Itih3','Cpb2','Pkm','Idh2','F13b','Gpld1','Alb','Krt10']

# Nodes chosen with confidence score
df = {'node':list_}
df = pd.DataFrame(data=df)

color_map_nodes_1 = ['seagreen','seagreen','chocolate','chocolate','mediumpurple','mediumpurple','deeppink','olivedrab','goldenrod','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow']
color_map_nodes_2 = ['seagreen','seagreen','chocolate','chocolate','mediumpurple','mediumpurple','deeppink','olivedrab','olivedrab','goldenrod','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow']
color_map_nodes_3 = ['seagreen','chocolate','mediumpurple','mediumpurple','deeppink','deeppink','olivedrab','olivedrab','goldenrod','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow']
color_map_nodes_4 = ['seagreen','seagreen','chocolate','mediumpurple','deeppink','deeppink','olivedrab','olivedrab','goldenrod','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow']
color_map_nodes_5 = ['seagreen','seagreen','chocolate','chocolate','mediumpurple','mediumpurple','deeppink','deeppink','olivedrab','goldenrod','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow']
color_map_nodes = ['seagreen','chocolate','mediumpurple','mediumpurple','deeppink','deeppink','olivedrab','goldenrod','goldenrod','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow','yellow']







# Filtering the string interaction database dataframe based on nodes chosen

df2 = df_1[(df_1.isin(list_).node1 & df_1.isin(list_).node2)]
# df2 = df_1[(df_1.isin(list_).node1 & df_1.isin(list_).node2) & (df_1.combined_score > 0.4)] # cannot use combine score filter, to use this modify the confidence score dictionary/dataframe as well




# Create graph
G = nx.from_pandas_edgelist(df2,'node1','node2', edge_attr='combined_score')

color_map_nodes_modified=[]
for v in G.nodes():
    u = np.where(v==df.node)[0][0]
    color_map_nodes_modified.append(color_map_nodes[u])



for v1,v2 in G.edges:
    G.edges[v1,v2]['feature'] = df_1.iloc[np.where(np.logical_or(np.logical_and (df_1.node1 == v1, df_1.node2 ==v2),np.logical_and (df_1.node1 == v2, df_1.node2 ==v1)))[0][0]]['combined_score']

# making dictionary of edge weights

edge_labels = dict([((n1, n2), d['feature'])
                    for n1, n2, d in G.edges(data=True)])


# Set position and color bar settings

layout = nx.spring_layout(G)


edges, weights = zip(*nx.get_edge_attributes(G, 'feature').items())
# Draw complete graph and the second draw just for edge with labels
nx.draw_networkx(G, pos=layout,  with_labels=True, node_color=color_map_nodes_modified , font_color='black',font_weight = 'bold',width =0.5,font_size=6,node_size = 500)



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
    "edge_cmap": plt.cm.turbo,
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
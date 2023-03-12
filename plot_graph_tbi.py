import config as cnf
import networkx as nx
import matplotlib.pyplot as plt

filepath = cnf.modelpath +'\TBI.pkl'
G = nx.read_gpickle(filepath)


pos = nx.random_layout(G)



dmin = 1
ncenter = 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if d < dmin:
        ncenter = n
        dmin = d

# color by path length from node near center
p = dict(nx.single_source_shortest_path_length(G, ncenter))

plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(p.keys()),
    node_size=80,
    node_color=list(p.values()),
    cmap=plt.cm.Reds_r,
)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis("off")
plt.show()

# nx.draw(g,with_labels=True)
# nx.draw_networkx(g,with_labels=True)
# plt.show()
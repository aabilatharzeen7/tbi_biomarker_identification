import dgl
import torch as th
import dgl
from dgl.data import DGLDataset
import torch
import networkx as nx
import config as cnf

class PLCgraphDataset(DGLDataset):

    def __init__(self):
        super().__init__(name='PLC graph')

    def process(self):

        filepath = cnf.modelpath + '\TBI.pkl'

        g = nx.read_gpickle(filepath)

        self.graph = dgl.from_networkx(g, node_attrs=['feature','label'])
        self.graph.ndata['feat'] = self.graph.ndata['feature']
        self.graph.ndata['label'] = self.graph.ndata['label']

        n_nodes = self.graph.num_nodes()
        n_train = int(n_nodes * 0.85)
        n_val = int(n_nodes * 0.15)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 1

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def load_plcgraph(filepath, train_ratio=0.005, valid_ratio=0.005):
    # load PLC data
    data = PLCgraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    return g, data.num_classes

# CHNAGES
def inductive_split(g):

    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""

    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g

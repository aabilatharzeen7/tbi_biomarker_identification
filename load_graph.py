import dgl
import numpy as np
import torch as th
import dgl
from dgl.data import DGLDataset
import torch
import networkx as nx
import config as cnf
import pandas as pd
import pickle
class PLCgraphDataset(DGLDataset):

    def __init__(self):
        super().__init__(name='PLC graph')

    def process(self):

        filepath = cnf.modelpath + '\TBI_t1.pkl'

        g = nx.read_gpickle(filepath)
        filepath = cnf.modelpath + 'modified_proteins'
        with open(filepath, 'rb') as f:
            modified_proteins= pickle.load(f)

        self.graph = dgl.from_networkx(g, node_attrs=['feature','label'])
        self.graph.ndata['feat'] = self.graph.ndata['feature']
        self.graph.ndata['label'] = self.graph.ndata['label']
        n_nodes= 149 # combined
        # n_nodes= 196 # round 2
        # n_nodes= 206  # round 1

        #
        # train_id, test_id = sk.train_test_split(range(self.graph.num_nodes()), test_size=0.3, shuffle=True,
        #                                         random_state=40)
        # train_id, val_id = sk.train_test_split(train_id, test_size=0.20, shuffle=True, random_state=40)
        #
        #
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)


        label = {'name': ['Apoe', 'Egfr', 'Clu', 'Grn', 'Vtn', 'Lrp1', 'Gsn', 'Reln', 'Mup12', 'Mup19', 'Mug1', 'Lifr',
                          'Itih1', 'Hgfac', 'Ubtfl1', 'Orm2', 'Spp2', 'Amy2a2'],
                 'value': [0.3125, 0.3125, 0.5, 0.375, 0.1875, 0.5625, 0.1875, 0.1875, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

        label_data = pd.DataFrame(data=label)
        train_list = label_data.sample(n=np.int(np.ceil(label_data.count() * 0.6)[1]), frac= None, replace=False, weights=None, random_state=None, axis=None)
        val_list = pd.concat([label_data,train_list]).drop_duplicates(keep=False)

        id = 0
        for u in self.graph.nodes():
            temp_id = u.numpy().item()
            temp_name = modified_proteins.iloc[np.where(temp_id==modified_proteins.id)[0],0].to_numpy()[0]
            if (train_list['name'].eq(temp_name)).any():
                train_mask[id] = True
            elif (val_list['name'].eq(temp_name)).any():
                val_mask[id] = True
            else:
                test_mask[id] = True

            id = id+1





        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        # n_nodes = self.graph.num_nodes()
        # n_train = int(np.ceil(n_nodes * 0.06))
        # n_val = int(np.ceil(n_nodes * 0.02))
        # train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        # val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        # test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #
        # train_mask[:n_train] = True
        # val_mask[n_train:n_train + n_val] = True
        # test_mask[n_train + n_val:] = True
        # self.graph.ndata['train_mask'] = train_mask
        # self.graph.ndata['val_mask'] = val_mask
        # self.graph.ndata['test_mask'] = test_mask

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 16

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
    val_g = g.subgraph(g.ndata['val_mask'])
    # val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g.subgraph(g.ndata['test_mask'])
    return train_g, val_g, test_g

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import config as cnf
import torch.optim as optim
import time
import argparse

from GNNmodel import SAGE
from load_graph import load_plcgraph, inductive_split

import pickle
import warnings
import shutil
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
warnings.filterwarnings("ignore")
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
from matplotlib import pyplot as plt







def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    th.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model):
    checkpoint = th.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, valid_loss_min

#### Entry point

def run(args, device, data,trained_model_path):

    # Unpack data

    n_classes, test_g, test_nfeat, test_labels = data

    in_feats = 10

    test_nid = th.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]

    dataloader_device = th.device('cpu')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])



    # define dataloader function
    def get_dataloader(train_g, train_nid, sampler):

        dataloader = dgl.dataloading.DataLoader(
            train_g,
            train_nid,
            sampler,
            device=dataloader_device,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)

        return dataloader

    dataloader = get_dataloader(test_g, test_nid, sampler)
    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)

    model = model.to(device)

    filepath = cnf.modelpath + 'modified_proteins'
    with open(filepath, 'rb') as f:
        modified_proteins = pickle.load(f)



    best_model= trained_model_path

    model, loss_min = load_ckp(best_model, model)
    model.eval()
    model = model.to(device)





    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        with th.no_grad():
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)



            pred=batch_pred


    chsn=10
    pred = pred.numpy()
    pred = pred.reshape(len(pred))
    pred = np.argsort(pred)

    ids = test_g.ndata['_ID']
    top_20_id = np.flip(ids[pred[-1*chsn:]].numpy())

    top_20_name = []
    for i in range(chsn):
        temp_id = top_20_id[i]
        top_20_name.append(modified_proteins.iloc[np.where(modified_proteins.id==temp_id)[0][0]]['name'])

    top_pred = batch_pred.numpy()
    top_pred = top_pred.reshape(len(top_pred))
    top_pred= np.sort(top_pred)
    print("Top protein",top_20_name)
    print("Values of protein",np.flip(top_pred[-1*chsn:]))

        # ans = batch_pred.numpy()
    plt.hist(pred)
    # plt.show()




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='PLC')
    argparser.add_argument('--num-epochs', type=int, default= 1)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='55,60,65')
    argparser.add_argument('--batch-size', type=int, default=196)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.15)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    fileext = "g6k"
    filepath = cnf.modelpath +'\TBI_t6.pkl'

    filepath = cnf.modelpath + 'modified_proteins'
    with open(filepath, 'rb') as f:
        modified_proteins = pickle.load(f)

    # changes
    if args.dataset == 'PLC':
        g, n_classes = load_plcgraph(filepath=filepath, train_ratio=0.75, valid_ratio=0.15)


    else:
        raise Exception('unknown dataset')

    # if args.inductive:
    train_g, val_g, test_g = inductive_split(g)


    test_nfeat = test_g.ndata.pop('features')
    test_labels = test_g.ndata.pop('labels')



    # Pack data
    data = n_classes, test_g,test_nfeat, test_labels
    trained_model_path = cnf.modelpath + "\\TBI_t6_trained_556065.pt"
    run(args, device, data,trained_model_path)



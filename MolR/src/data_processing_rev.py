import os
import dgl
from dgl.data import DGLDataset
import torch
import pickle
import pysmiles
from collections import defaultdict

class SmilesDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self, args, mode, raw_graph = None, feature_encoder = None):
        self.args = args
        self.mode = mode
        self.save_path = f'../data/USPTO-479k/cache/{self.args.data_type}_{self.args.split_type}/{self.mode}'
        self.raw_graph = raw_graph
        self.feature_encoder = feature_encoder
        self.reactant_graph = []
        self.product_graph = []
        super(MyDataset, self).__init__(name='Smiles_' + self.mode)

    def process(self):
        # process raw data to graphs, labels, splitting masks
        

    def __getitem__(self, idx):
        # get one example by index
        return self.reactant_graph[idx], self.product_graph[idx]

    def __len__(self):
        # number of data examples
        return len(self.reactant_graph)

    def save(self):
        # save processed data to directory `self.save_path`
        dgl.save_graphs(self.save_path + '_reactant_graphs.bin', self.reactant_graph)
        dgl.save_graphs(self.save_path + '_product_graphs.bin', self.product_graph)

    def load(self):
        # load processed data from directory `self.save_path`
        self.reactant_graph = dgl.load_graphs(self.save_path + '_reactant_graphs.bin', self.reactant_graph)[0]
        self.product_graph = dgl.load_graphs(self.save_path + '_product_graphs.bin', self.prodcut_graph)[0]
        self.to_gpu()

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        return os.path.exists(self.save_path + '_reactant_graphs.bin') and os.path.exists(self.save_path + '_product_graphs.bin')
    
    def to_gpu(self):
        if torch.cuda.is_available():
            self.reactant_graph = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graph]
            self.product_graph = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.product_graph]

def networkx_to_dgl(raw_graph, feature_encoder):
    # add edges
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    # add node features
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if raw_feature[j] in feature_encoder[j]:
                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]['unknown'])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata['feature'] = node_features
    # transform to bi-directed graph with self-loops
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    return graph
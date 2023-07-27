import os
import dgl
import torch
import pickle
import pysmiles
from collections import defaultdict
import logging
import networkx as nx

logger = logging.getLogger('pysmiles')
logger.setLevel(logging.ERROR)


attribute_names = ['element', 'charge', 'aromatic', 'hcount']

train_modes = ['train', 'train_sc']

class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, mode, feature_encoder=None, raw_graphs=None):
        self.args = args
        self.mode = mode
        self.feature_encoder = feature_encoder
        self.raw_graphs = raw_graphs
        self.path = f'../data/{self.args.dataset}/cache/{self.args.data_type}_{self.args.split_type}/{self.mode}'
        self.reactant_graphs = []
        self.product_graphs = []
        if self.mode == 'train':
            self.reactant_motifs = []
            self.product_motifs = []
        super().__init__(name='Smiles_' + mode)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.mode + ' data to GPU')
            self.reactant_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graphs]
            self.product_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.product_graphs]
            if self.mode == 'train':
                self.reactant_motifs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_motifs]
                self.product_motifs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.product_motifs]

    def save(self):
        print('saving ' + self.mode + ' reactants to ' + self.path + '_reactant_graphs.bin')
        print('saving ' + self.mode + ' products to ' + self.path + '_product_graphs.bin')
        dgl.save_graphs(self.path + '_reactant_graphs.bin', self.reactant_graphs)
        dgl.save_graphs(self.path + '_product_graphs.bin', self.product_graphs)
        if self.mode == 'train':
            dgl.save_graphs(self.path + '_reactant_motifs.bin', self.reactant_motifs)
            dgl.save_graphs(self.path + '_product_motifs.bin', self.product_motifs)

    def load(self):
        print('loading ' + self.mode + ' reactants from ' + self.path + '_reactant_graphs.bin')
        print('loading ' + self.mode + ' products from ' + self.path + '_product_graphs.bin')
        # graphs loaded from disk will have a default empty label set: [graphs, labels], so we only take the first item
        self.reactant_graphs = dgl.load_graphs(self.path + '_reactant_graphs.bin')[0]
        self.product_graphs = dgl.load_graphs(self.path + '_product_graphs.bin')[0]
        if self.mode == 'train':
            self.reactant_motifs = dgl.load_graphs(self.path + '_reactant_motifs.bin')[0]
            self.product_motifs = dgl.load_graphs(self.path + '_product_motifs.bin')[0]
        self.to_gpu()

    def process(self):
        print('transforming ' + self.mode + ' data from networkx graphs to DGL graphs')
        if self.mode == 'train':
            for i, (raw_reactant_graph, raw_product_graph, raw_reactant_motif, raw_product_motif) in enumerate(self.raw_graphs):
                if i % 10000 == 0:
                    print('%dk' % (i // 1000))
                # transform networkx graphs to dgl graphs
                reactant_graph = networkx_to_dgl(raw_reactant_graph, self.feature_encoder)
                product_graph = networkx_to_dgl(raw_product_graph, self.feature_encoder)
                self.reactant_graphs.append(reactant_graph)
                self.product_graphs.append(product_graph)
                reactant_motif = networkx_to_dgl(raw_reactant_motif, self.feature_encoder)
                product_motif = networkx_to_dgl(raw_product_motif, self.feature_encoder)
                self.reactant_motifs.append(reactant_motif)
                self.product_motifs.append(product_motif)
        else:
            for i, (raw_reactant_graph, raw_product_graph) in enumerate(self.raw_graphs):
                if i % 10000 == 0:
                    print('%dk' % (i // 1000))
                # transform networkx graphs to dgl graphs
                reactant_graph = networkx_to_dgl(raw_reactant_graph, self.feature_encoder)
                product_graph = networkx_to_dgl(raw_product_graph, self.feature_encoder)
                self.reactant_graphs.append(reactant_graph)
                self.product_graphs.append(product_graph)
        self.to_gpu()

    def has_cache(self):
        if self.mode == 'train':
            return all([os.path.exists(self.path + '_reactant_graphs.bin'), os.path.exists(self.path + '_product_graphs.bin'), os.path.exists(self.path + '_reactant_motifs.bin'), os.path.exists(self.path + '_product_motifs.bin')])
        else:
            return all([os.path.exists(self.path + '_reactant_graphs.bin'), os.path.exists(self.path + '_product_graphs.bin')])

    def __getitem__(self, i):
        if self.mode == 'train':
            return self.reactant_graphs[i], self.product_graphs[i], self.reactant_motifs[i], self.product_motifs[i]
        else:
            return self.reactant_graphs[i], self.product_graphs[i]

    def __len__(self):
        return len(self.reactant_graphs)


def networkx_to_dgl(raw_graph, feature_encoder):
    # add edges
    raw_graph = nx.convert_node_labels_to_integers(raw_graph)
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    # add node features
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if j in raw_feature and raw_feature[j] in feature_encoder[j]:
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


def read_data(dataset, mode, split_type, data_type):
    path = f'../data/{dataset}/{mode}_{split_type}.csv'
    print('preprocessing %s data from %s' % (mode, path))

    # saving all possible values of each attribute (only for training data)
    all_values = defaultdict(set)
    graphs = []

    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            if mode == 'train':
                idx, product_smiles, reactant_smiles, product_motifs, reactant_motifs = line.strip().split(',')
            else:
                idx, product_smiles, reactant_smiles, _ = line.strip().split(',')

            # skip the first line
            if len(idx) == 0:
                continue

            if int(i) % 10000 == 0:
                print('%dk' % (int(i) // 1000))
            
            if data_type == 'prpair' and mode == 'train':
                pass
            #     if '.' in reactant_smiles:
            #         reactant_smiles = reactant_smiles.split('.')
            #         product_smiles = len(reactant_smiles) * [product_smiles]
            #     else:
            #         reactant_smiles = [reactant_smiles]
            #         product_smiles = [product_smiles]

            #     # pysmiles.read_smiles() will raise a ValueError: "The atom [se] is malformatted" on USPTO-479k dataset.
            #     # This is because "Se" is in a aromatic ring, so in USPTO-479k, "Se" is transformed to "se" to satisfy
            #     # SMILES rules. But pysmiles does not treat "se" as a valid atom and raise a ValueError. To handle this
            #     # case, I transform all "se" to "Se" in USPTO-479k.
            #     for reactant_smile, product_smile in zip(reactant_smiles, product_smiles):
            #         if '[se]' in reactant_smile:
            #             reactant_smile = reactant_smile.replace('[se]', '[Se]')
            #         if '[se]' in product_smile:
            #             product_smile = product_smile.replace('[se]', '[Se]')

            #         # use pysmiles.read_smiles() to parse SMILES and get graph objects (in networkx format)
            #         reactant_graph = pysmiles.read_smiles(reactant_smile, zero_order_bonds=False)
            #         product_graph = pysmiles.read_smiles(product_smile, zero_order_bonds=False)

            #         if mode == 'train':
            #             # store all values
            #             for graph in [reactant_graph, product_graph]:
            #                 for attr in attribute_names:
            #                     for _, value in graph.nodes(data=attr):
            #                         all_values[attr].add(value)

            #         graphs.append([reactant_graph, product_graph])
            
            elif data_type == 'equiv':
                # pysmiles.read_smiles() will raise a ValueError: "The atom [se] is malformatted" on USPTO-479k dataset.
                # This is because "Se" is in a aromatic ring, so in USPTO-479k, "Se" is transformed to "se" to satisfy
                # SMILES rules. But pysmiles does not treat "se" as a valid atom and raise a ValueError. To handle this
                # case, I transform all "se" to "Se" in USPTO-479k.
                if '[se]' in reactant_smiles:
                    reactant_smiles = reactant_smiles.replace('[se]', '[Se]')
                if '[se]' in product_smiles:
                    product_smiles = product_smiles.replace('[se]', '[Se]')
                if mode == 'train':
                    if '[se]' in reactant_motifs:
                        reactant_motifs = reactant_motifs.replace('[se]', '[Se]')
                    if '[se]' in product_motifs:
                        product_motifs = product_motifs.replace('[se]', '[Se]')

                # use pysmiles.read_smiles() to parse SMILES and get graph objects (in networkx format)
                # print(product_smiles)
                # print(product_motifs)
                reactant_graph = pysmiles.read_smiles(reactant_smiles, zero_order_bonds=False)
                product_graph = pysmiles.read_smiles(product_smiles, zero_order_bonds=False)
                if mode == 'train':
                    try:
                        reactant_motif_graph = pysmiles.read_smiles(reactant_motifs, zero_order_bonds=False)
                        product_motif_graph = pysmiles.read_smiles(product_motifs, zero_order_bonds=False)
                    except ValueError as e:
                        pass
                        # print(product_smiles)
                        # print(product_motifs)
                        # print(reactant_smiles)
                        # print(reactant_motifs)
                        # print(e)
                        # print('Aromatic')
                        # print(nx.to_dict_of_dicts(reactant_motif_graph))
                        # print(nx.to_dict_of_dicts(product_motif_graph))
                    # print(nx.to_dict_of_dicts(reactant_motif_graph))
                    # print(nx.to_dict_of_dicts(product_motif_graph))

                if mode == 'train':
                # store all values
                    for graph in [reactant_graph, product_graph, reactant_motif_graph, product_motif_graph]:
                        for attr in attribute_names:
                            for _, value in graph.nodes(data=attr):
                                if value is None:
                                    continue
                                all_values[attr].add(value)

                    graphs.append([reactant_graph, product_graph, reactant_motif_graph, product_motif_graph])
                else:
                    graphs.append([reactant_graph, product_graph])


            else:
                raise ValueError

    if mode == 'train':
        return all_values, graphs
    else:
        return graphs


def get_feature_encoder(all_values):
    feature_encoder = {}
    idx = 0
    # key: attribute; values: all possible values of the attribute
    for key, values in all_values.items():
        feature_encoder[key] = {}
        for value in values:
            feature_encoder[key][value] = idx
            idx += 1
        # for each attribute, we add an "unknown" key to handle unknown values during inference
        feature_encoder[key]['unknown'] = idx
        idx += 1

    return feature_encoder


def preprocess(dataset, data_type, split_type):
    print('preprocessing %s dataset' % dataset)

    # read all data and get all values for attributes
    all_values, train_graphs = read_data(dataset, 'train', split_type, data_type)
    valid_graphs = read_data(dataset, 'valid', split_type, data_type)
    test_graphs = read_data(dataset, 'test', split_type, data_type)

    # get one-hot encoder for attribute values
    feature_encoder = get_feature_encoder(all_values)

    cache_folder = data_type + '_' + split_type + '/'
    
    # save feature encoder to disk
    path = '../data/' + dataset + '/cache/' + cache_folder + 'feature_encoder.pkl'
    print('saving feature encoder to %s' % path)
    with open(path, 'wb') as f:
        pickle.dump(feature_encoder, f)

    return feature_encoder, train_graphs, valid_graphs, test_graphs


def load_data(args):
    # if datasets are already cached, skip preprocessing
    cache_folder = args.data_type + '_' + args.split_type + '/'

    root_dir = f'../data/' + args.dataset + '/cache/' + cache_folder
    
    if os.path.exists(root_dir) and len(os.listdir(root_dir)) >= 7:
        path = '../data/' + args.dataset + '/cache/' + cache_folder + 'feature_encoder.pkl'
        print('cache found\nloading feature encoder from %s' % path)
        with open(path, 'rb') as f:
            feature_encoder = pickle.load(f)
        train_dataset = SmilesDataset(args, 'train')
        valid_dataset = SmilesDataset(args, 'valid')
        test_dataset = SmilesDataset(args, 'test')

    else:
        print('no cache found')
        path = '../data/' + args.dataset + '/cache/' + cache_folder
        print('creating directory: %s' % path)
        if not os.path.exists(path):
            os.mkdir(path)
        feature_encoder, train_graphs, valid_graphs, test_graphs = preprocess(args.dataset, args.data_type, args.split_type)
        train_dataset = SmilesDataset(args, 'train', feature_encoder, train_graphs)
        valid_dataset = SmilesDataset(args, 'valid', feature_encoder, valid_graphs)
        test_dataset = SmilesDataset(args, 'test', feature_encoder, test_graphs)

    return feature_encoder, train_dataset, valid_dataset, test_dataset

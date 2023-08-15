import torch
import pickle
import dgllife
import deepchem as dc
import pandas as pd
from model import GNN, DownstreamModel
from dgl.dataloading import GraphDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train(args, data):

    # Build model (encoder + downstream)
    path = '../saved/' + args.pretrained_model + '/'
    print('loading hyperparameters of pretrained model from ' + path + 'hparams.pkl')
    with open(path + 'hparams.pkl', 'rb') as f:
        hparams = pickle.load(f)

    print('loading pretrained model from ' + path + 'model.pt')
    gnn_encoder = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'])
    if torch.cuda.is_available():
        gnn_encoder.load_state_dict(torch.load(path + 'model.pt'))
        gnn_encoder = gnn_encoder.cuda(args.gpu)
    else:
        gnn_encoder.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))
    
    model = DownstreamModel(gnn_encoder, data.tasks, 3, hparams['dim'], 512, 0.1, 'class')
    criterion = torch.nn.BCELoss(reduction='none')
    encoder_params = gnn_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = torch.optim.Adam(encoder_params, 0.0001)
    head_opt = torch.optim.Adam(head_params, 0.001)

    # Split data with scaffolds (train, valid, test)

    splitter = dgllife.utils.ScaffoldSplitter()
    train_data, valid_data, test_data = splitter.train_val_test_split(data, scaffold_func='smiles')

    # Load GraphDataLoader and train

    train_loader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model.train()
    for graphs, labels, valids, smiles in train_loader:
        preds = model(graphs)
        loss = criterion(preds, labels)
        loss = torch.sum(loss * valids) / torch.sum(valids)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.zero_grad()
        head_opt.zero_grad

    total_pred = []
    total_label = []
    total_valid = []

    # Load GraphDataLoader and evaluate

    valid_loader = GraphDataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    test_loader = GraphDataLoader(test_data, batch_size=args.batch_size, shuffle=True)


    with torch.no_grad():
        model.eval()
        for graphs, labels, valids, smiles in dataloader:
            preds = model(graphs)
            all_features.append(graph_embeddings)
            all_labels.append(labels)
            all_smiles.append(smiles)
        all_features = torch.cat(all_features, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        # all_smiles = torch.cat(all_smiles, dim=0).cpu().numpy()
        all_smiles = [val for sublist in all_smiles for val in sublist]


    # print('splitting dataset')

    # print(len(all_features), len(all_labels), len(all_smiles))
    # print(all_features)
    # print(all_labels)
    # print(all_smiles)

    # df = pd.DataFrame({'y': all_labels, 'smiles': all_smiles})
    # df['w'] = 0
    # df['X'] = 0

    # dc_dataset = dc.data.DiskDataset.from_dataframe(df, X='X', y='y', w='w', ids='smiles')

    # splitter = dc.splits.ScaffoldSplitter()
    # tr, va, te = splitter.split(dc_dataset)

    # train_features = all_features[tr]
    # train_labels = all_labels[tr]
    # valid_features = all_features[va]
    # valid_labels = all_labels[va]
    # test_features = all_features[te]
    # test_labels = all_labels[te]

    # train_features = all_features[: int(0.8 * len(data))]
    # train_labels = all_labels[: int(0.8 * len(data))]
    # valid_features = all_features[int(0.8 * len(data)): int(0.9 * len(data))]
    # valid_labels = all_labels[int(0.8 * len(data)): int(0.9 * len(data))]
    # test_features = all_features[int(0.9 * len(data)):]
    # test_labels = all_labels[int(0.9 * len(data)):]

    # print('training the classification model\n')
    # pred_model = LogisticRegression(solver='liblinear')
    # pred_model.fit(train_features, train_labels)
    # run_classification(pred_model, 'train', train_features, train_labels)
    # run_classification(pred_model, 'valid', valid_features, valid_labels)
    # auc_test = run_classification(pred_model, 'test', test_features, test_labels)
    return auc_test


def run_classification(model, mode, features, labels):
    acc = model.score(features, labels)
    auc = roc_auc_score(labels, model.predict_proba(features)[:, 1])
    print('%s acc: %.4f   auc: %.4f' % (mode, acc, auc))
    return auc

def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res

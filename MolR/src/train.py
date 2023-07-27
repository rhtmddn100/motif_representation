import os
import torch
import pickle
import data_processing
import numpy as np
from model import GNN
from copy import deepcopy
from dgl.dataloading import GraphDataLoader

# torch.set_printoptions(profile="full", linewidth=100000, sci_mode=False)


def train(args, data):
    feature_encoder, train_data, valid_data, test_data = data
    feature_len = sum([len(feature_encoder[key]) for key in data_processing.attribute_names])
    model = GNN(args.gnn, args.layer, feature_len, args.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if torch.cuda.is_available():
        model = model.cuda(args.gpu)

    best_model_params = None
    best_val_mrr = 0
    print('start training\n')

    print('initial case:')
    model.eval()
    evaluate(model, 'valid', valid_data, args)
    evaluate(model, 'test', test_data, args)
    print()

    for i in range(args.epoch):
        print('epoch %d:' % i)

        # train
        model.train()
        total_loss = 0
        for reactant_graphs, product_graphs in train_dataloader:
            reactant_embeddings = model(reactant_graphs)
            product_embeddings = model(product_graphs)

            if args.dist_type == 'hyperbolic':
                reactant_embeddings = normalize(reactant_embeddings)
                product_embeddings = normalize(product_embeddings)
            
            if args.hn == 'hn' and args.data_type == 'prpair':
                hard_negatives = sample_hn(product_graphs)
                loss = calculate_loss_hn(reactant_embeddings, product_embeddings, hard_negatives, args)
            else:
                loss = calculate_loss(reactant_embeddings, product_embeddings, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        
        print(f'Total loss: {total_loss}')

        # evaluate on the validation set
        val_mrr = evaluate(model, 'valid', valid_data, args)
        evaluate(model, 'test', test_data, args)

        # save the best model
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())

        print()

    # evaluation on the test set
    print('final results on the test set:')
    model.load_state_dict(best_model_params)
    evaluate(model, 'test', test_data, args)
    print()

    # save the model, hyperparameters, and feature encoder to disk
    if args.save_model:
        if not os.path.exists('../saved/'):
            print('creating directory: ../saved/')
            os.mkdir('../saved/')

        directory = '../saved/%s_%d_%s_%s_%s_%s_%s' % (args.gnn, args.dim, args.dataset, args.data_type, args.loss_type, args.dist_type, args.split_type)
        # directory += '_' + time.strftime('%Y%m%d%H%M%S', time.localtime())
        if not os.path.exists(directory):
            os.mkdir(directory)

        print('saving the model to directory: %s' % directory)
        torch.save(best_model_params, directory + '/model.pt')
        with open(directory + '/hparams.pkl', 'wb') as f:
            hp_dict = {'gnn': args.gnn, 'layer': args.layer, 'feature_len': feature_len, 'dim': args.dim}
            pickle.dump(hp_dict, f)
        with open(directory + '/feature_enc.pkl', 'wb') as f:
            pickle.dump(feature_encoder, f)


def calculate_loss_hn(reactant_embeddings, product_embeddings, args):
    pass

def sample_hn(product_graphs, args):
    for graph in product_graphs:
        rxn_center_list = graph.ndata['rxn_center']
        for i in range(args.neg_size):
            start_nodes = np.where(rxn_center_list == 1)[0].tolist()
            

def calculate_loss(reactant_embeddings, product_embeddings, args):
    if args.dist_type == 'euclidean':
        dist = torch.cdist(reactant_embeddings, product_embeddings, p=2)
    elif args.dist_type == 'hyperbolic':
        eps = 1e-5
        r_sqnorm = torch.clamp(torch.sum(reactant_embeddings * reactant_embeddings, dim=-1), 0, 1 - eps)
        p_sqnorm = torch.clamp(torch.sum(product_embeddings * product_embeddings, dim=-1), 0, 1 - eps)
        sqdist = torch.pow(torch.cdist(reactant_embeddings, product_embeddings, p=2), 2)
        x = sqdist / torch.ger((1 - r_sqnorm), (1 - p_sqnorm)) * 2 + 1
        z = torch.sqrt(torch.pow(x, 2) - 1)
        dist = torch.log(x + z)
    else:
        raise ValueError
    
    pos = torch.diag(dist)
    mask = torch.eye(args.batch_size)
    if torch.cuda.is_available():
        mask = mask.cuda(args.gpu)
    
    if args.loss_type == 'margin':
        neg = (1 - mask) * dist + mask * args.margin
        neg = torch.relu(args.margin - neg)
        loss = torch.mean(pos) + torch.sum(neg) / args.batch_size / (args.batch_size - 1)
    elif args.loss_type == 'exponential':
        neg = (1 - mask) * dist
        neg = torch.exp(neg)
        loss = torch.mean(pos) - torch.log(torch.sum(neg)) / args.batch_size / (args.batch_size - 1)
    else:
        raise ValueError
    
    return loss


def evaluate(model, mode, data, args):
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        all_product_embeddings = []
        product_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        for _, product_graphs in product_dataloader:
            product_embeddings = model(product_graphs)
            all_product_embeddings.append(product_embeddings)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)
        # rank
        all_rankings = []
        reactant_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        i = 0
        for reactant_graphs, _ in reactant_dataloader:
            reactant_embeddings = model(reactant_graphs)
            ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batch_size, len(data))), dim=1)
            i += args.batch_size
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda(args.gpu)
            dist = torch.cdist(reactant_embeddings, all_product_embeddings, p=2)
            sorted_indices = torch.argsort(dist, dim=1)
            rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
            all_rankings.extend(rankings)

        # calculate metrics
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))

        print('%s  mrr: %.4f  mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f' % (mode, mrr, mr, h1, h3, h5, h10))
        return mrr

def normalize(e):
    return torch.renorm(e, 2, 0, 1)
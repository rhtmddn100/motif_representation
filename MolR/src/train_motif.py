import os
import torch
import pickle
import sys
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
    if args.gnn_shared != 'shared':
        model_motif = GNN(args.gnn, args.layer, feature_len, args.dim)
        optimizer_motif = torch.optim.Adam(model_motif.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if torch.cuda.is_available():
        model = model.cuda(args.gpu)
        if args.gnn_shared != 'shared':
            model_motif = model_motif.cuda(args.gpu)

    best_model_params = None
    best_val_mrr = 0
    best_loss = 10000000000000
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
        for reactant_graphs, product_graphs, reactant_motif_graphs, product_motif_graphs in train_dataloader:
            reactant_embeddings = model(reactant_graphs)
            product_embeddings = model(product_graphs)
            if args.gnn_shared == 'shared':
                reactant_motif_embeddings = model(reactant_motif_graphs)
                product_motif_embeddings = model(product_motif_graphs)
            else:
                reactant_motif_embeddings = model_motif(reactant_motif_graphs)
                product_motif_embeddings = model_motif(product_motif_graphs)

            # if args.dist_type == 'hyperbolic':
            #     reactant_embeddings = normalize(reactant_embeddings)
            #     product_embeddings = normalize(product_embeddings)
            
            loss = calculate_loss(reactant_embeddings, product_embeddings, reactant_motif_embeddings, product_motif_embeddings, args)

            optimizer.zero_grad()
            if args.gnn_shared != 'shared':
                optimizer_motif.zero_grad()
            loss.backward()
            optimizer.step()
            if args.gnn_shared != 'shared':
                optimizer_motif.step()
            total_loss += loss.item()
        
        print(f'Total loss: {total_loss}')

        # evaluate on the validation set
        val_mrr = evaluate(model, 'valid', valid_data, args)
        evaluate(model, 'test', test_data, args)

        # save the best model
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())

        # if total_loss < best_loss:
        #     best_loss = total_loss
        #     best_model_params = deepcopy(model.state_dict())

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

        directory = '../saved/%s_%s_%d_%s_%s_%s_%s_%s_%s_%s' % (args.dataset, args.gnn, args.dim, args.data_type, args.loss_type, args.dist_type, args.split_type, args.pretrain_type, 'gnn_'+args.gnn_shared, args.info)
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
        with open(directory + '/logs.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print(f'Best loss: {best_loss}')
            model.load_state_dict(best_model_params)
            evaluate(model, 'test', test_data, args)
            sys.stdout = original_stdout

            
def contrastive_loss(dist, args):
    pos = torch.diag(dist)
    mask = torch.eye(args.batch_size)
    if torch.cuda.is_available():
        mask = mask.cuda(args.gpu)
    
    neg = (1 - mask) * dist + mask * args.margin
    neg = torch.relu(args.margin - neg)

    return torch.mean(pos) + torch.sum(neg) / args.batch_size / (args.batch_size - 1)

def hyperbolic_dist(embedding1, embedding2, p=2):
    eps = 1e-5
    r_sqnorm = torch.clamp(torch.sum(embedding1 * embedding1, dim=-1), 0, 1 - eps)
    p_sqnorm = torch.clamp(torch.sum(embedding2 * embedding2, dim=-1), 0, 1 - eps)
    sqdist = torch.pow(torch.cdist(embedding1, embedding2, p=2), 2)
    x = sqdist / torch.ger((1 - r_sqnorm), (1 - p_sqnorm)) * 2 + 1
    z = torch.sqrt(torch.pow(x, 2) - 1)
    dist = torch.log(x + z)
    return dist

def calculate_loss(reactant_embeddings, product_embeddings, reactant_motif_embeddings, product_motif_embeddings, args):
    # if args.dist_type == 'euclidean':
    #     dist_func = torch.cdist
    # elif args.dist_type == 'hyperbolic':
    #     dist_func = hyperbolic_dist
    # else:
    #     raise ValueError

    if not args.pretrain_type == 'motif':
        dist_mols = torch.cdist(reactant_embeddings, product_embeddings, p=2)
        loss_mols = contrastive_loss(dist_mols, args)
    
    dist_reactant_motifs = torch.cdist(reactant_embeddings, reactant_motif_embeddings, p=2)
    dist_product_motifs = torch.cdist(product_embeddings, product_motif_embeddings, p=2)
    loss_reactant_motifs = contrastive_loss(dist_reactant_motifs, args)
    loss_product_motifs = contrastive_loss(dist_product_motifs, args)
    
    main_ratio = args.loss_ratio
    sub_ratio = (1 - main_ratio)/2

    if not args.pretrain_type == 'motif':
        return main_ratio*loss_mols + sub_ratio*loss_reactant_motifs + sub_ratio*loss_product_motifs
    else:
        return 0.5*loss_reactant_motifs + 0.5*loss_product_motifs


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
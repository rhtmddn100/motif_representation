import os
import argparse
import data_processing_motif
import train_motif
import numpy as np
from property_pred import pp_data_processing, pp_train
from ged_pred import gp_data_processing, gp_train
# from visualization import visualize


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')

    # '''
    # pretraining / chemical reaction prediction
    parser.add_argument('--task', type=str, default='pretrain', help='downstream task')
    parser.add_argument('--dataset', type=str, default='USPTO-479k', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--gnn', type=str, default='tag', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=1024, help='dimension of molecule embeddings')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--data_type', type=str, default='prpair', help='type of data to use')
    parser.add_argument('--dist_type', type=str, default='euclidean', help='type of distance metric to use')
    parser.add_argument('--loss_type', type=str, default='margin', help='type of loss to use')
    parser.add_argument('--split_type', type=str, default='random', help='type of split to use')
    parser.add_argument('--pretrain_type', type=str, default='mol_motif', help='type of pretraining to use')
    parser.add_argument('--loss_ratio', type=float, default=0.8, help='ratio of the main loss to subsidiary losses')
    parser.add_argument('--gnn_shared', type=str, default='shared', help='shared GNN between molecules and motifs')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model to disk')
    parser.add_argument('--info', type=str, default='', help='info for saving')
    # '''

    '''
    # molecule property prediction
    parser.add_argument('--task', type=str, default='property_pred', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='tag_1024', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='all', help='dataset name')
    parser.add_argument('--repeat', type=int, default=20, help='number of times to repeat the experiment')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for calling the pretrained model')
    '''

    '''
    # GED prediction
    parser.add_argument('--task', type=str, default='ged_pred', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='gcn_1024', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='QM9', help='dataset name')
    parser.add_argument('--n_molecules', type=int, default=1000, help='the number of molecules to be sampled')
    parser.add_argument('--n_pairs', type=int, default=10000, help='the number of molecule pairs to be sampled')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for calling the pretrained model')
    parser.add_argument('--feature_mode', type=str, default='concat', help='how to construct the input feature')
    '''

    '''
    # visualization
    parser.add_argument('--task', type=str, default='visualization', help='downstream task')
    parser.add_argument('--subtask', type=str, default='size', help='subtask type: reaction, property, ged, size, ring')
    parser.add_argument('--pretrained_model', type=str, default='gcn_1024', help='the pretrained model')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for calling the pretrained model')
    parser.add_argument('--dataset', type=str, default='BBBP', help='dataset name')
    '''

    args = parser.parse_args()
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    if args.task == 'pretrain':
        print('Start loading data')
        data = data_processing_motif.load_data(args)
        print('Finished loading data')
        train_motif.train(args, data)
    elif args.task == 'property_pred':
        if args.dataset == 'all':
            for dataset in ['BBBP', 'HIV', 'BACE', 'Tox21', 'ClinTox']:
                args.dataset = dataset
                test_aucs = []
                for _ in range(args.repeat):
                    data = pp_data_processing.load_data(args)
                    test_auc = pp_train.train(args, data)
                    test_aucs.append(test_auc)
                    print('---------------------------------------------------------------')
                print('Final Results:')
                print(f'{dataset} results with {args.repeat} runs: AVG {np.mean(test_aucs)} / STDEV {np.std(test_aucs)}')
                print('---------------------------------------------------------------')
        else:
            data = pp_data_processing.load_data(args)
            pp_train.train(args, data)
    elif args.task == 'ged_pred':
        data = gp_data_processing.load_data(args)
        gp_train.train(args, data)
    elif args.task == 'visualization':
        visualize.draw(args)
    else:
        raise ValueError('unknown task')


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1

print('Loading data')
uspto = pd.read_csv('../data/USPTO-50k/total.csv')

train_cutoff = frac_train * len(uspto)
valid_cutoff = (frac_train + frac_valid) * len(uspto)
train_inds = []
valid_inds = []
test_inds = []

print('Computing clusters')
sim_matrix = np.load('../data/Similarity/sim_matrix_all.npy')

for i in range(50016):
    sim_matrix[i][i] = 1.0

sim_matrix = 1 - sim_matrix

n_clusters = 40

clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
clustering.fit(sim_matrix)
clusters = clustering.labels_

print('Creating scaffold sets')
scaffold_sets = [np.where(clusters == i)[0].tolist() for i in range(n_clusters)]
scaffold_sets.sort(key=len, reverse = True)

print('\nSplitting indices')
for scaffold_set in scaffold_sets:
    if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
            test_inds += scaffold_set
        else:
            valid_inds += scaffold_set
    else:
        train_inds += scaffold_set

print('Creating csv for scaffold dataset')

uspto = uspto[['product', 'reactant', 'reactions']]
train_sc = uspto.iloc[train_inds].reset_index()
valid_sc = uspto.iloc[valid_inds].reset_index()
test_sc = uspto.iloc[test_inds].reset_index()

train_sc = train_sc[['product', 'reactant', 'reactions']]
valid_sc = valid_sc[['product', 'reactant', 'reactions']]
test_sc = test_sc[['product', 'reactant', 'reactions']]


train_sc.to_csv('../data/USPTO-50k/train_scaffold.csv')
valid_sc.to_csv('../data/USPTO-50k/valid_scaffold.csv')
test_sc.to_csv('../data/USPTO-50k/test_scaffold.csv')
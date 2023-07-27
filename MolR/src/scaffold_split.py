import json
import gzip
import hashlib
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

frac_train, frac_valid, frac_test = 14/16, 1/16, 1/16

print('Loading data')
templates = pd.read_json('../data/USPTO-479k/uspto.templates.json.gz', orient='records', compression='gzip')
uspto = pd.read_csv('../data/USPTO-479k/total.csv')
uspto = pd.concat([templates.set_index('reaction_id'), uspto.set_index('_id')], axis=1, join='inner')

train_cutoff = frac_train * len(uspto)
valid_cutoff = (frac_train + frac_valid) * len(uspto)
train_inds = []
valid_inds = []
test_inds = []

print('Creating scaffold sets')
scaffold_sets = [uspto.index[uspto['reaction_smarts'] == smarts].tolist() for smarts in tqdm(uspto['reaction_smarts'].unique())]
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

uspto = uspto[['prod_smiles', 'reactant_smiles', 'rxn_smiles']]
train_sc = uspto.iloc[train_inds]
valid_sc = uspto.iloc[valid_inds]
test_sc = uspto.iloc[test_inds]

train_sc.to_csv('../data/USPTO-479k/train_scaffold.csv')
valid_sc.to_csv('../data/USPTO-479k/valid_scaffold.csv')
test_sc.to_csv('../data/USPTO-479k/test_scaffold.csv')

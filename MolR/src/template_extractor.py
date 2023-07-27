import json
import gzip
import hashlib
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from rdchiral import template_extractor

def can_parse(rsmi):
    react, spec, prod = rsmi.split('>')
    if Chem.MolFromSmiles(react) and Chem.MolFromSmiles(prod):
        return True
    else:
        return False

uspto_train = pd.read_csv('../data/USPTO-479k/train.csv')
uspto_val = pd.read_csv('../data/USPTO-479k/valid.csv')
uspto_test = pd.read_csv('../data/USPTO-479k/test.csv')

uspto = pd.concat([uspto_train, uspto_val, uspto_test], ignore_index=True)

split_smiles = uspto['rxn_smiles'].str.split('>', expand=True)
uspto['reactants'] = split_smiles[0]
uspto['spectators'] = split_smiles[1]
uspto['products'] = split_smiles[2]

parsable = uspto['rxn_smiles'].map(can_parse)
uspto = uspto[parsable]

hexhash = (uspto['rxn_smiles']).apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())

uspto['source'] = 'uspto'
uspto['source_id'] = hexhash

uspto = uspto.reset_index().rename(columns={'index': '_id'})

uspto.to_csv('../data/USPTO-479k/total.csv')

reactions = uspto[['_id', 'reactants', 'products', 'spectators', 'source', 'source_id']]

reactions.to_json('../data/USPTO-479k/uspto.reactions.json.gz', orient='records', compression='gzip')

with gzip.open('../data/USPTO-479k/uspto.reactions.json.gz') as f:
    reactions = json.load(f)

def extract(reaction):
    try:
        return template_extractor.extract_from_reaction(reaction)
    except KeyboardInterrupt:
        print('Interrupted')
        raise KeyboardInterrupt
    except Exception as e:
        print(e)
        return {'reaction_id': reaction['_id']}

print("Start extracting templates")
templates = [extract(reaction) for reaction in tqdm(reactions)]

templates = pd.DataFrame(templates)

print("Saving templates")
templates.to_json('../data/USPTO-479k/uspto.templates.json.gz', orient='records', compression='gzip')

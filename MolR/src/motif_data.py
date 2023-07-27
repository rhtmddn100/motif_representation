import pandas as pd

data_dir = '../data/USPTO-479k'

uspto_train = pd.read_csv(data_dir + '/train_random.csv')
motif_lookup = pd.read_csv(data_dir + '/train_motifs.csv')

motif_data = {'prod_smiles': [], 'reactant_smiles': [], 'prod_motifs': [], 'reactant_motifs': []}

for index, row in uspto_train.iterrows():
    print(f'Processing line {index + 1}')
    prod = row['prod_smiles']
    reac = row['reactant_smiles']
    prods = prod.split('.')
    reacs = reac.split('.')
    
    p_motifs = []
    for p in prods:
        m = motif_lookup[motif_lookup['smiles'] == p]['motifs'].iloc[0]
        p_motifs.append(m)

    r_motifs = []
    for r in reacs:
        m = motif_lookup[motif_lookup['smiles'] == r]['motifs'].iloc[0]
        r_motifs.append(m)
    
    p_motif = '.'.join(p_motifs)
    r_motif = '.'.join(r_motifs)

    print(p_motif)
    print(r_motif)

    motif_data['prod_smiles'].append(prod)
    motif_data['reactant_smiles'].append(reac)
    motif_data['prod_motifs'].append(p_motif)
    motif_data['reactant_motifs'].append(r_motif)

motif_data_df = pd.DataFrame(motif_data)

motif_data_df.to_csv(data_dir + '/train_motif.csv')

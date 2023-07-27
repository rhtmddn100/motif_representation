import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FragmentOnBonds
import random
import re
import pickle

def get_mol_substructure(smiles, rxn_center, N=10):
    # Read molecule from smiles
    mol = Chem.MolFromSmiles(smiles)
    
    # Get all bonds
    bonds = mol.GetBonds()
    bond_list = []
    for bond in bonds:
        # Filter bonds (do not break rings, bonds connected to rxn center)
        if bond.GetBeginAtom().GetAtomMapNum() in rxn_center or bond.GetEndAtom().GetAtomMapNum in rxn_center:
            continue
        if bond.IsInRing():
            continue
        bond_list.append(bond.GetIdx())
    
    substructs = []
    # Sample random bonds equal to N
    rand_bond_list = []

    # Prepare dummy and H atoms for replacements
    dummy = Chem.MolFromSmiles('[*]')
    H = Chem.MolFromSmiles('[H]')
    
    # Create substructures
    while len(substructs) < N and len(bond_list) > 0:
        rand_bond = bond_list.pop(random.randrange(len(bond_list)))
        rand_bond_list.append(rand_bond)
        try:
            frag = FragmentOnBonds(mol, [rand_bond])
            frag = AllChem.ReplaceSubstructs(frag, dummy, H, replaceAll=True)[0]
            frag = Chem.RemoveHs(frag)
            frag_smiles = Chem.MolToSmiles(frag)
            # Split fragments into two and only select the part with reaction center
            frags = frag_smiles.split('.')
            for smi in frags:
                idx_list = re.findall(':(\d+)', smi)
                idx_list = [int(i) for i in idx_list]
                for center in rxn_center:
                    if center in idx_list:
                        substructs.append(smi)
                        break
        except:
            pass
                
    return substructs, rand_bond_list

def find_center(rxn):
    r, p = rxn.split('>>')
    rmol = Chem.MolFromSmiles(r)
    pmol = Chem.MolFromSmiles(p)
    rbonds = rmol.GetBonds()
    pbonds = pmol.GetBonds()

    rset = set()
    pset = set()
    for rbond in rbonds:
        rset.add((rbond.GetBeginAtom().GetAtomMapNum(), rbond.GetEndAtom().GetAtomMapNum()))

    for pbond in pbonds:
        pset.add((pbond.GetBeginAtom().GetAtomMapNum(), pbond.GetEndAtom().GetAtomMapNum()))

    cset = pset - rset
    center = set()
    for pair in cset:
        center.add(pair[0])
        center.add(pair[1])
    
    return center


for split in ['train', 'valid', 'test']:
    substructs_list = []
    rand_bond_list_list = []
    idx = 0
    with open(f'../data/USPTO-479k/{split}_random.csv', 'r') as f:
        for i, line in enumerate(f.readlines()):
            idx, _, _, rxn = line.strip().split(',')

            # skip the first line
            if len(idx) == 0:
                continue
            
            print(f'Processing {idx} for {split}')
            center = find_center(rxn)
            prod = rxn.split('>>')[-1]
            substructs, rand_bond_list = get_mol_substructure(prod, center)
            substructs_list.append(substructs)
            # rand_bond_list_list.append(rand_bond_list_list)

    with open(f'../data/Negative-Samples/{split}.pkl', 'wb') as fp:
        pickle.dump(substructs_list, fp)
    
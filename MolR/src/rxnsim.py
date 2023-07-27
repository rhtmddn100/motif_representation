import rdkit.Chem


def ____rxn_sim(reactions, reversible = True, algo = 'msim', sim_method = 'tanimoto', fp_type = 'extended', fp_mode = 'bit', fp_depth = 6, fp_size = 1024):
    algo = algo.lower()
    sim_method = sim_method.lower()
    fp_type = fp_type.lower()
    fp_mode = fp_mode.lower()

    if (reactions is None | len(reactions) < 2):
        print("Pass two or more reactions to compute similarity")
        return
    


def rxn_sim(rxnA, rxnB, z):
    
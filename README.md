# motif_representation

## 1. Extracting Motifs

1) Save the list of smiles that you want to fragment into motifs as a .smiles file and store under `MotifVocab/data` directory.

```
MotifVocab
├── data
│   └── QM9
│       ├── train.smiles
│       └── valid.smiles
|   └── USPTO-479k
│       ├── train.smiles
│       └── valid.smiles
├── output/
├── preprocess/
├── src/
└── README.md
```

2) Run the following command to obtain a csv file of all smiles, fragmented into motifs.

```
python MotifVocab/src/merging_operation_learning.py \
    --dataset USPTO-479k \
    --num_workers 60
```

## 2. Running Motif Representation Experiments

1) Unzip the `USPTO-479k.zip` file under `MolR/data/USPTO-479k` directory and rename each file to end with `_random.csv`. Additionally save the obtained csv file of all smiles fragmented into motifs in the same directory.

```
MolR
├── data
│   └── USPTO-479k
│       ├── train_random.csv
│       ├── valid_random.csv
│       ├── test_random.csv
│       └── train_motifs.csv
```

2) Run the following command to obtain a csv file for motif training.

```
python MolR/src/motif_data.py
```

3) Train and evaluate on motif representation using the following commands with desired parameters (refer to each file for details on the parameters).

Training:

```
python MolR/src/main_motifs.py
```

Evaluation for Property Prediction:

```
python MolR/src/main_motifs_pp.py
```


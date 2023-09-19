# motif_representation

## 1. Extracting Motifs

1) Save the list of smiles that you want to fragment into motifs as a .smiles file and store under `./data` directory.

```
MotifVocab
├── data
│   └── QM9
│       ├── train.smiles
│       └── valid.smiles
├── output/
├── preprocess/
├── src/
└── README.md
```

2) Run the following command to obtain a csv file of all smiles, fragmented into motifs

```
python src/merging_operation_learning.py \
    --dataset QM9 \
    --num_workers 60
```

## 2. 

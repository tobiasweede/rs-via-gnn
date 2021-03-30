# Recommender Systems via Graph Neural Networks

## Requirements

- dask (https://docs.dask.org/en/latest/)
- torchtext
- pandas

## Prepare datasets

### MovieLens 1M

1. Download http://files.grouplens.org/datasets/movielens/ml-1m.zip
2. Run `python process_movielens1m.py ./ml-1m ./data.pkl`

### Nowplaying-rs

1. Download https://zenodo.org/record/3248543/files/nowplayingrs.zip?download=1
2. Run `python preprocess_nowplaying_rs.py ./nowplaying_rs_dataset ./data.pkl`

## Run model

### Nearest-neighbor recommendation

```
python model.py data.pkl --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 64
```

### Sparse model

```
python model_sparse.py data.pkl --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 1024
```
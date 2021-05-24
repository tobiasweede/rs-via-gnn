# Graph Convolutional Matrix Completion

Paper link: [https://arxiv.org/abs/1706.02263](https://arxiv.org/abs/1706.02263)
Author's code: [https://github.com/riannevdberg/gc-mc](https://github.com/riannevdberg/gc-mc)

The implementation does not handle side-channel features and mini-epoching and thus achieves
slightly worse performance when using node features.

Credit: Jiani Zhang ([@jennyzhang0215](https://github.com/jennyzhang0215))

## Dependencies
* PyTorch 1.2+
* pandas
* torchtext 0.4+ (if using user and item contents as node features)
* spacy (if using user and item contents as node features)
    - You will also need to run `python -m spacy download en_core_web_sm`

## Data

Supported datasets: amazon

## How to run
### Train with full-graph
```bash
python3 train.py --data_name=electronic --use_one_hot_fea --gcn_agg_accum=stack --gcn_out_units=100
```

IGMC -- Inductive Graph-based Matrix Completion
===============================================================================

Credits
-----
* https://github.com/muhanzhang/IGMC
> M. Zhang and Y. Chen, Inductive Matrix Completion Based on Graph Neural Networks. [\[PDF\]](https://openreview.net/pdf?id=ByxxgCEYDS)

Usages
------

### Flixster, Douban and YahooMusic

To train on Flixster, type:

    python Main.py --data-name flixster --epochs 40 --testing --ensemble

The results will be saved in "results/flixster\_testmode/". The processed enclosing subgraphs will be saved in "data/flixster/testmode/". Change flixster to douban or yahoo\_music to do the same experiments on Douban and YahooMusic datasets, respectively. Delete --testing to evaluate on a validation set to do hyperparameter tuning.

### MovieLens-100K and MovieLens-1M

To train on MovieLens-100K, type:

    python Main.py --data-name ml_100k --save-appendix _mnph200 --data-appendix _mnph200 --epochs 80 --max-nodes-per-hop 200 --testing --ensemble --dynamic-train

where the --max-nodes-per-hop argument specifies the maximum number of neighbors to sample for each node during the enclosing subgraph extraction, whose purpose is to limit the subgraph size to accomodate large datasets. The --dynamic-train option makes the training enclosing subgraphs dynamically generated rather than generated in a preprocessing step and saved in disk, whose purpose is to reduce memory consumption. However, you may remove the option to generate a static dataset for future reuses. Append "--dynamic-test" to make the test dataset also dynamic. The default batch size is 50, if a batch cannot fit into your GPU memory, you can reduce batch size by appending "--batch-size 25" to the above command.

The results will be saved in "results/ml\_100k\_mnph200\_testmode/". The processed enclosing subgraphs will be saved in "data/ml\_100k\_mnph200/testmode/" if you do not use dynamic datasets. 

To train on MovieLens-1M, type:
    
    python Main.py --data-name ml_1m --save-appendix _mnhp100 --data-appendix _mnph100 --max-nodes-per-hop 100 --testing --epochs 40 --save-interval 5 --adj-dropout 0 --lr-decay-step-size 20 --ensemble --dynamic-train
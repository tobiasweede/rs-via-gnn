# RS via GNN

## Overview
This repository contains experiment code for the 
master thesis ***Recommender Systems via Graph Neural Networks***.

We address:
* Fundamental proof of concept for the _Message Passing_ paradigm
* MLN baselines for different RS data sets
* GNN experiments for different RS data sets

## Structure
_Only relevant files and folders are listed:_
* __00_zachary__
  * XXXX.ipynb: All the code lies here.
* __01_mln-baselines__
  * XXXX.ipynb: All the code lies here.
* __02_gcmc__
  * XXXX.ipynb: All the code lies here.
* __03_igmc__
  * XXXX.ipynb: All the code lies here.
    
## Environment / Installation

It is suggested to use a dedicated python instance.
We use _conda_. Required packages can be found in `environment.yml`.
See [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for further instructions.

## Credits
For our experiments we heavily rely on other people's work.
We thank everyone who made her research and code publicly available.
The original sources are marked in the individual files.

Here is the list of GNN frameworks (which we suggest as a starting point for others interested in crafting GNNs):
* https://github.com/dmlc/dgl/
* https://github.com/rusty1s/pytorch_geometric
* https://github.com/danielegrattarola/spektral
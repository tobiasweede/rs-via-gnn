#!/usr/bin/env python3
"""
Script that reads from raw MovieLens-100k data and dumps into a pickle
file the following:

* A heterogeneous graph with categorical features.
* A list with all the movie titles.  The movie titles correspond to
  the movie nodes in the heterogeneous graph.
"""

import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path

    ## Build heterogeneous graph

    # Load data

    ratings = []
    with open(os.path.join(directory, 'u.data'), encoding='utf-8') as f:
        for l in f:
            user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('\t')]
            ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': timestamp,
                })
    ratings = pd.DataFrame(ratings)

    users = []
    for user_id in ratings['user_id'].unique():
        users.append({
            'user_id': user_id
        })
    users = pd.DataFrame(users)

    movies = []
    for movie_id in ratings['movie_id'].unique():
        movies.append({
            'movie_id': movie_id
        })
    movies = pd.DataFrame(movies)

    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'user_id', 'user')
    graph_builder.add_entities(movies, 'movie_id', 'movie')
    graph_builder.add_binary_relations(ratings, 'user_id', 'movie_id', 'watched')
    graph_builder.add_binary_relations(ratings, 'movie_id', 'user_id', 'watched-by')

    g = graph_builder.build()

    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.
    g.edges['watched'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['watched'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
    g.edges['watched-by'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)

    # Train-validation-test split
    train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'user_id')


    # Build the graph with training interactions only.
    train_g = build_train_graph(g, train_indices, 'user', 'movie', 'watched', 'watched-by')
    assert train_g.out_degrees(etype='watched').min() > 0

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'movie', 'watched')

    ## Dump the graph and the datasets

    dataset = {
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': None,
        'item-images': None,
        'user-type': 'user',
        'item-type': 'movie',
        'user-to-item-type': 'watched',
        'item-to-user-type': 'watched-by',
        'timestamp-edge-column': 'timestamp'}


    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

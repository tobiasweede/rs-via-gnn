"""MovieLens dataset"""
import os
import zipfile

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch as th
from dgl.data.utils import get_download_dir
from kaggle.api.kaggle_api_extended import KaggleApi  # for data download

from utils import to_etype_name

class Goodreads(object):
    """Goodreads dataset used by GCMC model

    The dataset stores Amazon ratings in two types of graphs.

    The encoder graph contains rating value information in the form of edge types.

    The decoder graph stores plain user-item pairs in the form of a bipartite graph with no rating
    information. All graphs have two types of nodes: "user" and "item".

    The training, validation and test set can be summarized as follows:

    training_enc_graph : training user-item pairs + rating info
    training_dec_graph : training user-item pairs
    valid_enc_graph : training user-item pairs + rating info
    valid_dec_graph : validation user-item pairs
    test_enc_graph : training user-item pairs + validation user-item pairs + rating info
    test_dec_graph : test user-item pairs

    Attributes
    ----------
    train_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for training.
    train_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for training.
    train_labels : torch.Tensor
        The categorical label of each user-item pair
    train_truths : torch.Tensor
        The actual rating values of each user-item pair
    valid_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for validation.
    valid_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for validation.
    valid_labels : torch.Tensor
        The categorical label of each user-item pair
    valid_truths : torch.Tensor
        The actual rating values of each user-item pair
    test_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for test.
    test_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for test.
    test_labels : torch.Tensor
        The categorical label of each user-item pair
    test_truths : torch.Tensor
        The actual rating values of each user-item pair
    user_feature : torch.Tensor
        User feature tensor. If None, representing an identity matrix.
    item_feature : torch.Tensor
        Movie feature tensor. If None, representing an identity matrix.
    possible_rating_values : np.ndarray
        Available rating values in the dataset

    Parameters
    ----------
    name : str
        Dataset name. Could be "electronic"
    device : torch.device
        Device context
    mix_cpu_gpu : boo, optional
        If true, the ``user_feature`` attribute is stored in CPU
    use_one_hot_fea : bool, optional
        If true, the ``user_feature`` attribute is None, representing an one-hot identity
        matrix. (Default: False)
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    test_ratio : float, optional
        Ratio of test data
    valid_ratio : float, optional
        Ratio of validation data

    """

    def __init__(self, name, device, mix_cpu_gpu=False,
                 use_one_hot_fea=False, symm=True,
                 test_ratio=0.2, valid_ratio=0.2):
        self._name = name
        self._device = device
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        self._dir = '/home/weiss/rs_data/goodreads/'
        self._ratingsfile = 'goodreads_interactions.csv'
        print("Starting processing {} ...".format(self._name))
        print('......')
        if self._name == 'goodreads':
            self.all_rating_info = self._load_raw_rates(os.path.join(self._dir, self._ratingsfile), sep=',')
            self._load_raw_user_info()
            self._load_raw_item_info()
            num_test = int(np.ceil(self.all_rating_info.shape[0] * self._test_ratio))
            shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
            self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
            self.all_train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test:]]
        else:
            raise NotImplementedError
        print('......')
        num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
        shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
        self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
        self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid:]]
        self.possible_rating_values = np.unique(self.train_rating_info["rating"].values)

        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tAll train rating pairs : {}".format(self.all_train_rating_info.shape[0]))
        print("\t\tTrain rating pairs : {}".format(self.train_rating_info.shape[0]))
        print("\t\tValid rating pairs : {}".format(self.valid_rating_info.shape[0]))
        print("\tTest rating pairs  : {}".format(self.test_rating_info.shape[0]))

        # self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
        #                                          cmp_col_name="id",
        #                                          reserved_ids_set=set(self.all_rating_info["user_id"].values),
        #                                          label="user")

        # self.item_info = self._drop_unseen_nodes(orign_info=self.item_info,
        #                                           cmp_col_name="id",
        #                                           reserved_ids_set=set(self.all_rating_info["item_id"].values),
        #                                           label="item")

        # Map user/item to the global id
        self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        self.global_item_id_map = {ele: i for i, ele in enumerate(self.item_info['id'])}
        print('Total user number = {}, item number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_item_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_item = len(self.global_item_id_map)

        ### Generate features
        if use_one_hot_fea:
            self.user_feature = None
            self.item_feature = None
        else:
            # if mix_cpu_gpu, we put features in CPU
            # if mix_cpu_gpu:
            #     self.user_feature = th.FloatTensor(self._process_user_fea())
            #     self.item_feature = th.FloatTensor(self._process_item_fea())
            # else:
            #     self.user_feature = th.FloatTensor(self._process_user_fea()).to(self._device)
            #     self.item_feature = th.FloatTensor(self._process_item_fea()).to(self._device)
            raise NotImplementedError
        if self.user_feature is None:
            self.user_feature_shape = (self.num_user, self.num_user)
            self.item_feature_shape = (self.num_item, self.num_item)
        else:
            self.user_feature_shape = self.user_feature.shape
            self.item_feature_shape = self.item_feature.shape
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nitem: {}".format(self.item_feature_shape)
        print(info_line)

        all_train_rating_pairs, all_train_rating_values = self._generate_pair_value(self.all_train_rating_info)
        train_rating_pairs, train_rating_values = self._generate_pair_value(self.train_rating_info)
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(self.valid_rating_info)
        test_rating_pairs, test_rating_values = self._generate_pair_value(self.test_rating_info)

        def _make_labels(ratings):
            labels = th.LongTensor(np.searchsorted(self.possible_rating_values, ratings)).to(device)
            return labels

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values, add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs)
        self.train_labels = _make_labels(train_rating_values)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self.valid_labels = _make_labels(valid_rating_values)
        self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

        self.test_enc_graph = self._generate_enc_graph(all_train_rating_pairs, all_train_rating_values,
                                                       add_support=True)
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self.test_labels = _make_labels(test_rating_values)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.number_of_edges(str(r))
            return rst

        print("Train enc graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('item'),
            _npairs(self.train_enc_graph)))
        print("Train dec graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
            self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('item'),
            self.train_dec_graph.number_of_edges()))
        print("Valid enc graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('item'),
            _npairs(self.valid_enc_graph)))
        print("Valid dec graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
            self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('item'),
            self.valid_dec_graph.number_of_edges()))
        print("Test enc graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('item'),
            _npairs(self.test_enc_graph)))
        print("Test dec graph: \t#user:{}\t#item:{}\t#pairs:{}".format(
            self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('item'),
            self.test_dec_graph.number_of_edges()))

    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_item_id_map[ele] for ele in rating_info["item_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        user_item_R = np.zeros((self._num_user, self._num_item), dtype=np.float32)
        user_item_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'item': self._num_item}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'item'): (rrow, rcol),
                ('item', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            user_ci = []
            user_cj = []
            item_ci = []
            item_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                item_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    item_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    item_cj.append(th.zeros((self.num_item,)))
            user_ci = _calc_norm(sum(user_ci))
            item_ci = _calc_norm(sum(item_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                item_cj = _calc_norm(sum(item_cj))
            else:
                user_cj = th.ones(self.num_user, )
                item_cj = th.ones(self.num_item, )
            graph.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
            graph.nodes['item'].data.update({'ci': item_ci, 'cj': item_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])  # add one indicating edge for every (user, item) pair
        user_item_ratings_coo = sp.coo_matrix(  # create coo matrix for graph conversion
            (ones, rating_pairs),
            shape=(self.num_user, self.num_item), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_item_ratings_coo, utype='_U', etype='_E', vtype='_V')
        return dgl.heterograph({('user', 'rate', 'item'): g.edges()},
                               num_nodes_dict={'user': self.num_user, 'item': self.num_item})

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_item

    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        print("  -----------------")
        print("{}: {} (reserved) vs. {} (from info)".format(label, len(reserved_ids_set),
                                                             len(set(orign_info[cmp_col_name].values))))
        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            print("\torign_info: ({}, {})".format(orign_info.shape[0], orign_info.shape[1]))
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            print("\tAfter dropping, data shape: ({}, {})".format(data_info.shape[0], data_info.shape[1]))
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info

    def _load_raw_rates(self, file_path, sep, n=50, m=1000):
        """In Goodreads, the rates have the following format
        user_id;book_id;is_read;rating;is_reviewed

        timestamp is unix timestamp and can be converted by pd.to_datetime(X, unit='s')

        Parameters
        ----------
        file_path : str

        Returns
        -------
        rating_info : pd.DataFrame
        """
        rating_info = pd.read_csv(self._dir + self._ratingsfile)
        rating_info = rating_info[rating_info['rating'] != 0]  # drop empty reviews
        rating_info.drop(columns=['is_read', 'is_reviewed'], inplace=True)
        rating_info.rename(columns={'book_id': 'item_id'}, inplace=True)

        # only keep users with more than n ratings
        counts = rating_info['user_id'].value_counts()  # count ratings per user
        mask = (counts >= n) & (counts <= m)
        rating_info = rating_info[rating_info['user_id'].isin(counts[counts >= n].index)]
        print(mask.value_counts())
        ratings = rating_info[rating_info['user_id'].isin(mask[mask == True].index)]

        return rating_info

    def _load_raw_user_info(self):
        """ For electronic, there is no user information. We read the user id from the rating file.

        Returns
        -------
        user_info : pd.DataFrame
        """
        rating_info = self.all_rating_info
        self.user_info = pd.DataFrame(np.unique(rating_info['user_id'].values), columns=['id'])

    def _load_raw_item_info(self):
        """ For electronic, there is no item information. We read the item id from the rating file.

        Returns
        -------
        user_info : pd.DataFrame
        """
        rating_info = self.all_rating_info
        self.item_info = pd.DataFrame(np.unique(rating_info['item_id'].values), columns=['id'])
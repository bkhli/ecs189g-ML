'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from local_code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
import random

from collections import defaultdict

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx
    
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1]) # this should be the y



        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # different normalization functions work better for different datasets
        # maybe this can be adjusted
        norm_adj = self.normalize(adj + sp.eye(adj.shape[0]))
        if self.dataset_name == 'cora':
            norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))
            

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        #features = self.normalize(features + sp.eye(features.shape))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train = []
        idx_test = []

        # same as what was in the other file
        np.random.seed(2) 


        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # keeping these in Dataset Loader
        if self.dataset_name == 'cora':
            idx_feature_mapping = {(1, 0, 0, 0, 0, 0, 0) : [],
                                (0, 1, 0, 0, 0, 0, 0) : [],
                                (0, 0, 1, 0, 0, 0, 0) : [],
                                (0, 0, 0, 1, 0, 0, 0) : [],
                                (0, 0, 0, 0, 1, 0, 0) : [],
                                (0, 0, 0, 0, 0, 1, 0) : [],
                                (0, 0, 0, 0, 0, 0, 1) : []}
            for i in range(features.shape[0]):
                idx_feature_mapping[tuple(onehot_labels[i])].append(i)
            for key in idx_feature_mapping:
                sampled_set = set(random.sample(idx_feature_mapping[key], 20))
                complement = set(idx_feature_mapping[key]) - sampled_set
                idx_train.extend(sampled_set)
                idx_test.extend(complement)
            
        elif self.dataset_name == 'citeseer':
            idx_feature_mapping = {(1, 0, 0, 0, 0, 0) : [],
                                (0, 1, 0, 0, 0, 0) : [],
                                (0, 0, 1, 0, 0, 0) : [],
                                (0, 0, 0, 1, 0, 0) : [],
                                (0, 0, 0, 0, 1, 0) : [],
                                (0, 0, 0, 0, 0, 1) : []}
            for i in range(features.shape[0]):
                idx_feature_mapping[tuple(onehot_labels[i])].append(i)
            for key in idx_feature_mapping:
                sampled_set = set(random.sample(idx_feature_mapping[key], 20))
                complement = set(idx_feature_mapping[key]) - sampled_set
                idx_train.extend(sampled_set)
                idx_test.extend(complement)
        elif self.dataset_name == 'pubmed':
            idx_feature_mapping = {(1, 0, 0) : [],
                                (0, 1, 0) : [],
                                (0, 0, 1) : []}
            for i in range(features.shape[0]):
                idx_feature_mapping[tuple(onehot_labels[i])].append(i)
            for key in idx_feature_mapping:
                sampled_set = set(random.sample(idx_feature_mapping[key], 20))
                complement = set(idx_feature_mapping[key]) - sampled_set
                idx_train.extend(sampled_set)
                idx_test.extend(complement)
        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)

        print("Printing idx train")
        print(idx_train)

        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)

        train_test = {'idx_train': idx_train, 'idx_test': idx_test}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test': train_test}
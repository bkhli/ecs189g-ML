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
        print("PRINTING SIZE")
        print(idx_features_labels)
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1]) # this should be the y
        print(onehot_labels)

        idx_feature_mapping = {(1, 0, 0, 0, 0, 0, 0) : [],
                               (0, 1, 0, 0, 0, 0, 0) : [],
                               (0, 0, 1, 0, 0, 0, 0) : [],
                               (0, 0, 0, 1, 0, 0, 0) : [],
                               (0, 0, 0, 0, 1, 0, 0) : [],
                               (0, 0, 0, 0, 0, 1, 0) : [],
                               (0, 0, 0, 0, 0, 0, 1) : []}
        


#         for input_item, one_hot_label in data:
#             output_dict[output_label].append(input_item)

#         # Convert to regular dict if needed
# o       utput_dict = dict(output_dict)
        print("Should be printing indices of nodes")
        print(idx_features_labels[:, 0])

        print("Should be printing length of features")
        print(features.shape[0])


        for i in range(features.shape[0]):
            idx_feature_mapping[tuple(onehot_labels[i])].append(i)

        print("Printing feature mapping")
        print(idx_feature_mapping)


        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # Still need to randomly sample these, also account for when there's different cases
        idx_train = []
        idx_test = []
        np.random.seed(2)

        for key in idx_feature_mapping:
            # print("printing i")
            # print(i)
           # sampled_set = np.random.choice(idx_feature_mapping[i], size=20, replace=False)
            sampled_set = set(random.sample(idx_feature_mapping[key], 20))
            complement = set(idx_feature_mapping[key]) - sampled_set
            idx_train+=sampled_set
            idx_test+=complement

        # idx_train = range(140)
        # idx_test = range(140, 140+1050)
        idx_train = torch.LongTensor(idx_train)
        #idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)






        train_test = {'idx_train': idx_train, 'idx_test': idx_test}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test': train_test}

        # graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        # return graph
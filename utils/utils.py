# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import random

import dgl
import errno
import json
import os
import torch
import torch.nn.functional as F
import numpy as np

from .dataset import MolReactionCSVDataset
from dgllife.utils import SMILESToBigraph, ScaffoldSplitter, RandomSplitter, ConsecutiveSplitter

def init_featurizer(args):
    """Initialize node/edge featurizer

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with featurizers updated
    """
    if args['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                         'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
        args['atom_featurizer_type'] = 'pre_train'
        args['bond_featurizer_type'] = 'pre_train'
        args['node_featurizer'] = PretrainAtomFeaturizer()
        args['edge_featurizer'] = PretrainBondFeaturizer()
        return args

    if args['atom_featurizer_type'] == 'canonical':
        from dgllife.utils import CanonicalAtomFeaturizer
        args['node_featurizer'] = CanonicalAtomFeaturizer()
    elif args['atom_featurizer_type'] == 'attentivefp':
        from dgllife.utils import AttentiveFPAtomFeaturizer
        args['node_featurizer'] = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], "
            "got {}".format(args['atom_featurizer_type']))

    if args['model'] in ['Weave', 'MPNN', 'AttentiveFP']:
        if args['bond_featurizer_type'] == 'canonical':
            from dgllife.utils import CanonicalBondFeaturizer
            args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
        elif args['bond_featurizer_type'] == 'attentivefp':
            from dgllife.utils import AttentiveFPBondFeaturizer
            args['edge_featurizer'] = AttentiveFPBondFeaturizer(self_loop=True)
    else:
        args['edge_featurizer'] = None

    return args

def load_dataset(args, df):
    smiles_to_g = SMILESToBigraph(add_self_loop=True, node_featurizer=args['node_featurizer'],
                                  edge_featurizer=args['edge_featurizer'])
    dataset = MolReactionCSVDataset(df=df,
                                    smiles_to_graph=smiles_to_g,
                                    smiles_column_a=args['smiles_column_1'],
                                    smiles_column_b=args['smiles_column_2'],
                                    cache_file_path=args['result_path'] + '/graph.bin',
                                    task_names=args['task_names'],
                                    n_jobs=args['num_workers'],
                                    load=False)
    return dataset

def get_configure(model):
    """Query for the manually specified configuration

    Parameters
    ----------
    model : str
        Model type

    Returns
    -------
    dict
        Returns the manually specified configuration
    """
    with open('model_zoos/configures/{}.json'.format(model), 'r') as f:
        config = json.load(f)
    return config

def mkdir_p(path):
    """Create a folder for the given path.

    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def init_trial_path(args):
    """Initialize the path for a hyperparameter setting

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with the trial path updated
    """
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = args['result_path'] + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args['trial_path'] = path_to_results
    mkdir_p(args['trial_path'])

    return args

def split_dataset(args, dataset):
    """Split the dataset

    Parameters
    ----------
    args : dict
        Settings
    dataset
        Dataset instance

    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    train_ratio, val_ratio, test_ratio = map(float, args['split_ratio'].split(','))
    if args['split'] == 'scaffold_decompose':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='decompose')
    elif args['split'] == 'scaffold_smiles':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args['split'] == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    elif args['split'] == 'ordinal':
        # shuffle point area data sort
        # train_end_ind = math.floor(len(dataset) * train_ratio)
        # val_end_ind = train_end_ind + math.floor(len(dataset) * val_ratio)
        # random.shuffle(dataset[0:train_end_ind])
        # random.shuffle(dataset[train_end_ind, val_end_ind])
        # random.shuffle(dataset[val_end_ind:])
        train_set, val_set, test_set = ConsecutiveSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    elif args['split'] == 'cross':
        folds = args['cross_split_folds']
        dataset_split_list = ConsecutiveSplitter.k_fold_split(
            dataset, folds)
        return dataset_split_list
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args['split']))

    return train_set, val_set, test_set

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))
    # construct two bg  np.array(graphs).flatten().tolist()
    graphs = np.array(graphs)
    bg_a = dgl.batch(graphs[:, 0])
    bg_a.set_n_initializer(dgl.init.zero_initializer)
    bg_a.set_e_initializer(dgl.init.zero_initializer)

    bg_b = dgl.batch(graphs[:, 1])
    bg_b.set_n_initializer(dgl.init.zero_initializer)
    bg_b.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, (bg_a, bg_b), labels, masks

def collate_molgraphs_unlabeled(data):
    """Batching a list of datapoints without labels

    Parameters
    ----------
    data : list of 2-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES and a DGLGraph.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    """
    # smiles, graphs = map(list, zip(*data))
    # bg = dgl.batch(graphs)
    # bg.set_n_initializer(dgl.init.zero_initializer)
    # bg.set_e_initializer(dgl.init.zero_initializer)
    smiles, graphs, labels, masks = map(list, zip(*data))
    # construct two bg  np.array(graphs).flatten().tolist()
    graphs = np.array(graphs)
    bg_a = dgl.batch(graphs[:, 0])
    bg_a.set_n_initializer(dgl.init.zero_initializer)
    bg_a.set_e_initializer(dgl.init.zero_initializer)

    bg_b = dgl.batch(graphs[:, 1])
    bg_b.set_n_initializer(dgl.init.zero_initializer)
    bg_b.set_e_initializer(dgl.init.zero_initializer)

    return smiles, (bg_a, bg_b), labels

def load_model(exp_configure):
    if exp_configure['model'] == 'DGAT':
        from model_zoos import DGATPredictor
        model = DGATPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            num_heads=[exp_configure['num_heads']] * exp_configure['num_gnn_layers'],
            feat_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            attn_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            alphas=[exp_configure['alpha']] * exp_configure['num_gnn_layers'],
            residuals=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            agg_modes=['mean'] * exp_configure['num_gnn_layers'],
            predictor_out_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']
        )
    else:
        return ValueError("Expect model to be from ['DGAT'], "
                          "got {}".format(exp_configure['model']))

    return model

def predict(args, model, bg):
    bg = (bg[0].to(args['device']), bg[1].to(args['device']))
    if args['edge_featurizer'] is None:
        node_feats_a = bg[0].ndata.pop('h').to(args['device'])
        node_feats_b = bg[1].ndata.pop('h').to(args['device'])
        return model(bg, (node_feats_a, node_feats_b))
    elif args['bond_featurizer_type'] == 'pre_train':
        node_feats = [
            bg.ndata.pop('atomic_number').to(args['device']),
            bg.ndata.pop('chirality_type').to(args['device'])
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(args['device']),
            bg.edata.pop('bond_direction_type').to(args['device'])
        ]
        return model(bg, node_feats, edge_feats)
    else:
        node_feats_a = bg[0].ndata.pop('h').to(args['device'])
        edge_feats_a = bg[0].edata.pop('e').to(args['device'])

        node_feats_b = bg[1].ndata.pop('h').to(args['device'])
        edge_feats_b = bg[1].edata.pop('e').to(args['device'])
        return model(bg, (node_feats_a, node_feats_b), (edge_feats_a, edge_feats_b))

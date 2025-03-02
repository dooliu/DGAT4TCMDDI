# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hyperopt import hp

common_hyperparameters = {
    'lr': hp.uniform('lr', low=1e-4, high=3e-1),
    'weight_decay': hp.uniform('weight_decay', low=0, high=3e-3),
    'patience': hp.choice('patience', [50]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512]),
}

dgat_hyperparameters = {
    'batch_size': hp.choice('batch_size', [128, 256, 512]),
    'gnn_hidden_feats': hp.choice('gnn_hidden_feats', [64, 128, 256, 512]),
    'num_heads': hp.choice('num_heads', [4, 6, 8]),
    'alpha': hp.uniform('alpha', low=0., high=0.5),
    'predictor_hidden_feats': hp.choice('predictor_hidden_feats', [32, 64, 128, 256]),
    'num_gnn_layers': hp.choice('num_gnn_layers', [2, 3, 4]),
    'residual': hp.choice('residual', [True, False]),
    'dropout': hp.uniform('dropout', low=0., high=0.6)
}

def init_hyper_space(model):
    """Initialize the hyperparameter search space

    Parameters
    ----------
    model : str
        Model for searching hyperparameters

    Returns
    -------
    dict
        Mapping hyperparameter names to the associated search spaces
    """
    candidate_hypers = dict()
    candidate_hypers.update(common_hyperparameters)
    if model == 'DGAT':
        candidate_hypers.update(dgat_hyperparameters)
    else:
        return ValueError('Unexpected model: {}'.format(model))
    return candidate_hypers

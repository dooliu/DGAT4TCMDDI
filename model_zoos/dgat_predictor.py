# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GATv2-based model for regression and classification on graphs
#
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn

from dgllife.model.model_zoo.mlp_predictor import MLPPredictor
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
import torch

from model_zoos.layers.gat_layer import GATMI, GATGraphLayer

import numpy as np
# pylint: disable=W0221
class DGATPredictor(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_feats=None,
        num_heads=None,
        feat_drops=None,
        attn_drops=None,
        alphas=None,
        residuals=None,
        activations=None,
        allow_zero_in_degree=False,
        biases=None,
        share_weights=None,
        agg_modes=None,
        n_tasks=1,
        predictor_out_feats=128,
        predictor_dropout=0.):
        super(DGATPredictor, self).__init__()

        self.gnn1 = GATMI(in_feats=in_feats,
                         hidden_feats=hidden_feats,
                         num_heads=num_heads,
                         feat_drops=feat_drops,
                         attn_drops=attn_drops,
                         alphas=alphas,
                         residuals=residuals,
                         activations=activations,
                         allow_zero_in_degree=allow_zero_in_degree,
                         biases=biases,
                         share_weights=share_weights,
                         agg_modes=agg_modes)

        self.gnn2 = GATMI(in_feats=in_feats,
                          hidden_feats=hidden_feats,
                          num_heads=num_heads,
                          feat_drops=feat_drops,
                          attn_drops=attn_drops,
                          alphas=alphas,
                          residuals=residuals,
                          activations=activations,
                          allow_zero_in_degree=allow_zero_in_degree,
                          biases=biases,
                          share_weights=share_weights,
                          agg_modes=agg_modes)

        if agg_modes[-1] == 'flatten':
            gnn_out_feats = hidden_feats[-1] * num_heads[-1]
        else:
            gnn_out_feats = hidden_feats[-1]
        #
        self.graph_conv1 = GATGraphLayer(in_feats=gnn_out_feats,
                                          num_heads=num_heads[-1],
                                          )

        self.graph_conv2 = GATGraphLayer(in_feats=gnn_out_feats,
                                          num_heads=num_heads[-1],
                                          )
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_out_feats,
                                    n_tasks, predictor_dropout)
        # self.predict = nn.Sequential(
        #     nn.RNN(input_size=2 * gnn_out_feats, hidden_size=predictor_out_feats, num_layers=2),
        #     nn.ReLU(),
        #     nn.Linear(predictor_out_feats, n_tasks)
        # )

    def forward(self, bg, feats, get_attention=False):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph tuple
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs.
            * M1 is the input node feature size, which equals in_feats in
              initialization
        get_attention : bool, optional
            Whether to return the attention values. Defaults: False

        Returns
        -------
        preds : FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        attentions : list of FloatTensor of shape (E, H, 1), optional
            It is returned when :attr:`get_attention` is True.
            ``attentions[i]`` gives the attention values in the i-th GATv2
            layer.

            * `E` is the number of edges.
            * `H` is the number of attention heads.
        """
        # print(f"input feats: {feats.shape}")
        if get_attention:
            node_feats, attentions = self.gnn1(bg, feats, get_attention=get_attention)
            graph_feats = self.graph_conv(bg, node_feats)
            print(f"gnn output shape:{graph_feats.shape}")
            print(f"gnn output:{graph_feats}")
            return self.predict(graph_feats), attentions
        else:
            node_feats_1 = self.gnn1(bg[0], feats[0])
            node_feats_2 = self.gnn2(bg[1], feats[1])
            graph_feats_1 = self.graph_conv1(bg[0], node_feats_1)
            graph_feats_2 = self.graph_conv2(bg[1], node_feats_2)

            # return self.predict(torch.cat((graph_feats_1, graph_feats_2), 1))
            return self.predict(graph_feats_1 + graph_feats_2)

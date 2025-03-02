# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Attention Networks v2
#
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv
from dgl.nn.pytorch import WeightAndSum

__all__ = ["GATNodeLayer", "GATGraphLayer"]

# pylint: disable=W0221
class GATNodeLayer(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        share_weights=False,
        agg_mode="flatten",
    ):
        super(GATNodeLayer, self).__init__()
        self.gatv2_conv = GATv2Conv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
            bias=bias,
            share_weights=share_weights,
        )
        assert agg_mode in ["flatten", "mean"]
        self.agg_mode = agg_mode

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gatv2_conv.reset_parameters()

    def forward(self, bg, feats, get_attention=False):
        """Update node representations

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs.
            * M1 is the input node feature size, which equals in_feats in
              initialization
        get_attention : bool, optional
            Whether to return the attention values. Defaults: False

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs.
            * M2 is the output node representation size, which equals
              out_feats in initialization if self.agg_mode == 'mean' and
              out_feats * num_heads otherwise.
        attention : FloatTensor of shape (E, H, 1), optional
            Attention values, returned when :attr:`get_attention` is True

            * `E` is the number of edges.
            * `H` is the number of attention heads.
        """
        if get_attention:
            out_feats, attention = self.gatv2_conv(
                bg, feats, get_attention=True
            )
        else:
            out_feats = self.gatv2_conv(bg, feats)

        if self.agg_mode == "flatten":
            out_feats = out_feats.flatten(1)
        else:
            out_feats = out_feats.mean(1)

        if get_attention:
            return out_feats, attention
        else:
            return out_feats


# pylint: disable=W0221
class GATGraphLayer(nn.Module):
    r"""Single GATv2 layer from `How Attentive Are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`
    """

    def __init__(
        self,
        in_feats,
        num_heads,
    ):
        super(GATGraphLayer, self).__init__()
        print(f"num_heads:{num_heads}")
        self.graph_conv_heads = nn.ModuleList()
        for i in range(num_heads):
            self.graph_conv_heads.append(WeightAndSum(in_feats))
        # self.weight_and_sum =
        # assert agg_mode in ["flatten", "mean"]
        # self.agg_mode = agg_mode

        # self.feat_drop = nn.Dropout(feat_drop)
        # self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, bg, feats):
        """Readout

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        graph_feature : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        graph_features = []
        for i in range(len(self.graph_conv_heads)):
            h_g_sum = self.graph_conv_heads[i](bg, feats)
            with bg.local_scope():
                bg.ndata['h'] = feats
                h_g_max = dgl.max_nodes(bg, 'h')
            features = torch.cat([h_g_sum, h_g_max], dim=1)
            graph_features.append(features)
            # print(f"graph_features type:{type(features)}, {features.shape}")
        # print(f"before:{graph_features[0].shape}, {graph_features}")
        graph_features = torch.stack(graph_features)
        # print(f"after:{graph_features.shape}, {graph_features}")
        # print(f"mean res: {graph_features.mean(0).shape},{graph_features.mean(0)}")
        return graph_features.mean(0)

class GATMI(nn.Module):
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
    ):
        super(GATMI, self).__init__()

        if hidden_feats is None:
            hidden_feats = [32, 32]

        n_layers = len(hidden_feats)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0.0 for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0.0 for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [False for _ in range(n_layers)]
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        if share_weights is None:
            share_weights = [False for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ["flatten" for _ in range(n_layers - 1)]
            agg_modes.append("mean")

        lengths = [
            len(hidden_feats),
            len(num_heads),
            len(feat_drops),
            len(attn_drops),
            len(alphas),
            len(residuals),
            len(activations),
            len(biases),
            len(share_weights),
            len(agg_modes),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of hidden_feats, num_heads, feat_drops, "
            "attn_drops, alphas, residuals, activations, biases, "
            "share_weights, and agg_modes to be the same, "
            "got {}".format(lengths)
        )
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GATNodeLayer(
                    in_feats=in_feats,
                    out_feats=hidden_feats[i],
                    num_heads=num_heads[i],
                    feat_drop=feat_drops[i],
                    attn_drop=attn_drops[i],
                    negative_slope=alphas[i],
                    residual=residuals[i],
                    activation=activations[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                    bias=biases[i],
                    share_weights=share_weights[i],
                    agg_mode=agg_modes[i],
                )
            )
            if agg_modes[i] == "flatten":
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, bg, feats, get_attention=False):
        """Update node representations.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs.
            * M1 is the input node feature size, which equals in_feats in
              initialization
        get_attention : bool, optional
            Whether to return the attention values. Defaults: False

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs.
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] if agg_modes[-1] == 'mean' and
              hidden_sizes[-1] * num_heads[-1] otherwise.
        attentions : list of FloatTensor of shape (E, H, 1), optional
            It is returned when :attr:`get_attention` is True.
            ``attentions[i]`` gives the attention values in the i-th GATv2
            layer.

            * `E` is the number of edges.
            * `H` is the number of attention heads.
        """
        if get_attention:
            attentions = []
            for gnn in self.gnn_layers:
                feats, attention = gnn(bg, feats, get_attention=get_attention)
                attentions.append(attention)
            return feats, attentions
        else:
            for gnn in self.gnn_layers:
                feats = gnn(bg, feats, get_attention=get_attention)
            return feats

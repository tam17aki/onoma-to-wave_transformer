# -*- coding: utf-8 -*-
"""Component modules for Onoma-to-Wave with Transformer.

Copyright (C) 2022 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


# adapted from
# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py
class BatchNormConv1d(nn.Module):
    """BatchNorm with Conv1d.

    Args:
        dimentions (tuple): tuple of input and output channel sizes
        conv_config (tuple): tuple of kernel size, stride, and padding size
        activation (torch.nn.Module or None): non-linear activation function

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """

    def __init__(self, dimentions, conv_config, activation=None):
        """Initialize class."""
        super().__init__()
        input_dim = dimentions[0]
        output_dim = dimentions[1]
        kernel_size = conv_config[0]
        stride = conv_config[1]
        padding = conv_config[2]
        self.conv1d = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(output_dim)
        self.activation = activation

    def forward(self, inputs):
        """Forward propagation."""
        outputs = self.conv1d(inputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return self.batchnorm(outputs)


# adapted from
# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py
class HighwayNet(nn.Module):
    """Highway Network.

    Args:
        input_size (int): input channel sizes
        output_size (int): output channel sizes

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """

    def __init__(self, input_size, output_size):
        """Initialize class."""
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear.bias.data.zero_()
        self.gate = nn.Linear(input_size, output_size)
        self.gate.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """Forward propagation."""
        highway = self.relu(self.linear(inputs))
        gate = self.sigmoid(self.gate(inputs))
        return highway * gate + inputs * (1.0 - gate)


class ScaledPositionalEncoding(nn.Module):
    """Module for scaled positional encoding.

    Args:
        emb_size (int): embedding size of transformer (= feature size of self-attention)
        dropout (float): dropout rate
        maxlen (int): maximum size of encoding positionmu

    See Sec. 3.2 of 'Neural Speech Synthesis with Transformer Network'.
    https://arxiv.org/abs/1809.08895
    """

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 1000):
        """Initialize class."""
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size)
        )
        position = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)  # [1, maxlen, emb_size]

        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(1.0))  # to be optimized
        self.register_buffer("pos_embedding", pos_embedding)  # make self.pos_embedding

    def forward(self, inputs):
        """Forward propagation."""
        outputs = inputs + self.scale * self.pos_embedding[:, : inputs.size(1), :]
        return self.dropout(outputs)


class EncoderPrenetConfig(NamedTuple):
    """Class for Encoder Prenet configuration.

    Args:
        padding_idx (int): padding index for onomatopoeia.
        input_dim (int): number of phonemes.
        embed_dim (int): embedding dimension of phonemes.
        attention_dim (int): embedding dimension of Transformer.
        conv_channels (int): number of channels in convolution.
        kernel_size (int): size of convolutional kernel.
        n_layers (int): number of layers.
        dropout (float): dropout rate.
    """

    padding_idx: int
    input_dim: int
    embed_dim: int
    attention_dim: int
    conv_channels: int = 256
    kernel_size: int = 3
    n_layers: int = 2
    dropout: float = 0.5


# adapted from
# https://github.com/r9y9/ttslearn/blob/master/ttslearn/tacotron/encoder.py
class EncoderPrenet(nn.Module):
    """Encoder PreNet Class.

    Args:
        config (EncoderPrenetConfig):
            input_dim (int): number of phonemes
            embed_dim (int): embedded dimension
            attention_dim (int): attention dimension in Transformer
            conv_channels (int): number of hidden layers
            kernel_size (int): number of hidden layers
            layers (int): number of hidden layers
            dropout (float): number of hidden layers

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """

    def __init__(self, config):
        """Initialize class."""
        super().__init__()
        input_dim = config.input_dim
        embed_dim = config.embed_dim
        attention_dim = config.attention_dim
        conv_channels = config.conv_channels
        kernel_size = config.kernel_size
        n_layers = config.n_layers
        dropout = config.dropout
        self.embedding = None
        self.prenet = None
        self.projection = None

        if n_layers != 0:
            self.embedding = nn.Embedding(
                input_dim, embed_dim, padding_idx=config.padding_idx
            )
            convs = nn.ModuleList()
            for layer in range(n_layers):
                in_channels = embed_dim if layer == 0 else conv_channels
                convs += [
                    BatchNormConv1d(
                        (in_channels, conv_channels),
                        (kernel_size, 1, (kernel_size - 1) // 2),
                        nn.ReLU(),
                    ),
                    nn.Dropout(dropout),
                ]
            self.prenet = nn.Sequential(*convs)
            self.projection = nn.Linear(conv_channels, attention_dim)
        else:
            self.embedding = nn.Embedding(
                input_dim, attention_dim, padding_idx=config.padding_idx
            )

    def forward(self, inputs):
        """Forward propagation."""
        outputs = self.embedding(inputs)  # [batch_size, src_len, emb_dim]
        if self.prenet is not None:
            outputs = outputs.transpose(1, 2)  # [B, L, C] -> [B, C, L] for conv1d
            for layer in self.prenet:
                outputs = layer(outputs)
            outputs = outputs.transpose(1, 2)
            outputs = self.projection(outputs)
        return outputs


class DecoderPrenetConfig(NamedTuple):
    """Class for Decoder Prenet configuration.

    Args:
        input_dim (int): number of mel-bands (feature dim. of mel-spectrogram).
        attention_dim (int): embedding dimension of Transformer.
        n_units (int): number of dimentionality in fully-connected layers.
        n_layers (int): number of layers.
    """

    input_dim: int
    attention_dim: int
    n_units: int = 256
    n_layers: int = 2
    dropout: float = 0.5


# adapted from
# https://github.com/r9y9/ttslearn/blob/master/ttslearn/tacotron/decoder.py
class DecoderPrenet(nn.Module):
    """Decoder PreNet class.

    Args:
        config (DecoderPrenetConfig):
            input_dim (int): number of mel-bands (#feature dim. of mel-spectrogram).
            attention_dim (int): attention dimension in Transformer.
            n_units (int): embedded dimension
            n_layers  (int): number of hidden layers
            dropout  (float): number of hidden layers

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """

    def __init__(self, config):
        """Initialize class."""
        super().__init__()
        input_dim = config.input_dim
        attention_dim = config.attention_dim
        n_units = config.n_units
        n_layers = config.n_layers
        self.dropout = config.dropout
        self.prenet = None
        self.projection = None

        if n_layers != 0:
            layers = nn.ModuleList()
            for layer in range(n_layers):
                layers += [
                    nn.Linear(input_dim if layer == 0 else n_units, n_units),
                    nn.ReLU(),
                ]
            self.prenet = nn.Sequential(*layers)
            self.projection = nn.Linear(n_units, attention_dim)
        else:
            self.projection = nn.Linear(input_dim, attention_dim)

    def forward(self, inputs):
        """Forward propagation."""
        outputs = inputs
        if self.prenet is not None:
            for layer in self.prenet:
                # dropout is applied in both training and inference
                outputs = F.dropout(layer(outputs), self.dropout, training=True)
        outputs = self.projection(outputs)
        return outputs


class PostnetConfig(NamedTuple):
    """Class for Postnet configuration.

    Args:
        input_dim (int): number of mel-bands (#feature dim. of mel-spectrogram).
        conv_channels (int): number of channels in convolution.
        kernel_size (int): size of convolutional kernel.
        n_layers (int): number of layers.
        dropout (float): dropout rate.
    """

    input_dim: int
    conv_channels: int = 512
    kernel_size: int = 5
    n_layers: int = 5
    dropout: float = 0.1


# adapted from
# https://github.com/r9y9/ttslearn/blob/master/ttslearn/tacotron/postnet.py
class Postnet(nn.Module):
    """Post-Net module for Transformer.

    This modules is utilized to refine the prediction of mel-spectrogram.

    Args:
        config (PostnetConfig):
            input_dim (int): number of mel-bands (#feature dim. of mel-spectrogram).
            conv_channels (int): number of channels.
            kernel_size (int): kernel size.
            n_layers (int): number of layers.
            dropout (float): dropout rate.

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """

    def __init__(self, config):
        """Initialize class."""
        super().__init__()
        in_dim = config.input_dim
        n_layers = config.n_layers
        conv_channels = config.conv_channels
        kernel_size = config.kernel_size
        dropout = config.dropout
        postnet = nn.ModuleList()
        for layer in range(n_layers):
            in_channels = in_dim if layer == 0 else conv_channels
            out_channels = in_dim if layer == n_layers - 1 else conv_channels
            postnet += [
                BatchNormConv1d(
                    (in_channels, out_channels),
                    (kernel_size, 1, (kernel_size - 1) // 2),
                    activation=None,
                )
            ]
            if layer != n_layers - 1:
                postnet += [nn.Tanh()]
            postnet += [nn.Dropout(dropout)]
        self.postnet = nn.Sequential(*postnet)

    def forward(self, inputs):
        """Forward step.

        Args:
            inputs (torch.Tensor): input sequence
        Returns:
            torch.Tensor: output sequence
        """
        inputs = inputs.transpose(1, 2)
        outputs = self.postnet(inputs)
        outputs = outputs.transpose(1, 2)
        return outputs


class EventEmbedding(nn.Module):
    """Embed one-hot event tensors."""

    def __init__(self, input_dim, output_dim):
        """Initialize single embedding layer."""
        super().__init__()
        self.fc_layer = nn.Linear(input_dim, output_dim)

    def forward(self, inputs, event_tensor):
        """Perform forward propagation."""
        event_tensor = event_tensor.unsqueeze(1).repeat(1, inputs.shape[1], 1)
        inputs = torch.cat([inputs, event_tensor], dim=2)
        outputs = self.fc_layer(inputs)
        return outputs


class CBHGConfig(NamedTuple):
    """Class for CBHG configuration.

    Args:
        input_dim (int): number of convolution channels
        n_convbanks (int): number of convolution banks.
        n_highways (int): number of layers in highway networks.
        projections (tuple): list of projection dims.
    """

    input_dim: int
    n_convbanks: int = 8
    n_highways: int = 4
    projections: tuple = 512


# adapted from
# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py
class CBHG(nn.Module):
    """CBHG: convert mel-spectrogram into linear-spectrogram.

    Args:
        config (CBHGConfig):
            input_dim (int): number of mel-bands (#feature dim of mel-spectrogram).
            n_convbanks (int): number of convolution banks.
            n_highways (int): number of layers in highway networks.
            projections (tuple): list of projection dims.
        device (torch.device): "cuda" or "cpu".

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """

    def __init__(self, config, device):
        """Initialize class."""
        super().__init__()
        input_dim = config.input_dim
        n_convbanks = config.n_convbanks
        n_highways = config.n_highways
        projections = list(config.projections)
        self.input_dim = input_dim
        self.conv1d_banks = nn.ModuleList(
            [
                BatchNormConv1d((input_dim, input_dim), (k, 1, k // 2), nn.ReLU())
                for k in range(1, n_convbanks + 1)
            ]
        )
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [n_convbanks * input_dim] + projections[:-1]
        activations = [nn.ReLU()] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [
                BatchNormConv1d((in_size, out_size), (3, 1, 1), activation=ac)
                for (in_size, out_size, ac) in zip(in_sizes, projections, activations)
            ]
        )
        self.linear_proj = nn.Linear(
            projections[-1], input_dim, bias=False, device=device
        )
        self.highways = nn.ModuleList(
            [HighwayNet(input_dim, input_dim) for _ in range(n_highways)]
        )
        self.gru = nn.GRU(input_dim, input_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        """Forward propagation."""
        # (B, T_in, input_dim)
        hidden = inputs

        # Needed to perform conv1d on time-axis
        # (B, input_dim, T_in)
        if hidden.size(-1) == self.input_dim:
            hidden = hidden.transpose(1, 2)

        n_frame = hidden.size(-1)

        # (B, input_dim*K, T_in)
        # Concat conv1d bank outputs
        hidden = torch.cat(
            [conv1d(hidden)[:, :, :n_frame] for conv1d in self.conv1d_banks], dim=1
        )
        assert hidden.size(1) == self.input_dim * len(self.conv1d_banks)
        hidden = self.max_pool1d(hidden)[:, :, :n_frame]

        for conv1d in self.conv1d_projections:
            hidden = conv1d(hidden)

        # (B, T_in, input_dim)
        # Back to the original shape
        hidden = hidden.transpose(1, 2)

        if hidden.size(-1) != self.input_dim:
            hidden = self.linear_proj.forward(hidden)

        # Residual connection
        hidden += inputs
        for highway in self.highways:
            hidden = highway(hidden)

        if input_lengths is not None:
            hidden = nn.utils.rnn.pack_padded_sequence(
                hidden, input_lengths, batch_first=True, enforce_sorted=False
            )

        # (B, T_in, input_dim*2)
        outputs, _ = self.gru(hidden)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

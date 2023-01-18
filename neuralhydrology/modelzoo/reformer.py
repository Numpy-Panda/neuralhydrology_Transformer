import logging
import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.Encoder import EncoderLayer, Encoder, ReformerLayer
from neuralhydrology.modelzoo.time_features_embedding import time_features, DataEmbedding ##################

LOGGER = logging.getLogger(__name__)


class Reformer(BaseModel):
    """Reformer model class.

    The model configuration is specified in the config file using the following options:+
    
    * ``reformer_layers``: number of reformer encoder layers.
    * ``reformer_nheads``: number of attention heads.
    * ``reformer_bucket_size``: hash bucket size
    * ``reformer_n_hashes``: number of hash.
    * ``reformer_dropout``: reformer dropout.


    Parameters
    ----------
    cfg : Config
        The run configuration.
        
            
    References
    ----------
    .. [#] Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451.
    """
    
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super(Reformer, self).__init__(cfg=cfg)
        
        # encoder specific
        n_heads = cfg.reformer_nheads
        e_layers = cfg.reformer_layers
        bucket_size = cfg.reformer_bucket_size
        n_hashes = cfg.reformer_n_hashes
        dropout = cfg.reformer_dropout
        activation = 'gelu'
        d_ff = None
        print('********************')
        print('You are using Reformer')
        print(f'number of reformer encoder layers: {int(e_layers)}')
        print(f'number of attention heads : {int(n_heads)}')
        print(f'hash bucket size: {int(bucket_size)}')
        print(f'number of hashes: {int(n_hashes)}')
        print(f'reformer dropout: {dropout}')
        print('********************')

        # embedding net before transformer
        self.embedding_net = InputLayer(cfg)

        # ensure that the number of inputs into the self-attention layer is divisible by the number of heads
        if self.embedding_net.output_size % cfg.transformer_nheads != 0:
            raise ValueError("Embedding dimension must be divisible by number of transformer heads. "
                             "Use statics_embedding/dynamics_embedding and embedding_hiddens to specify the embedding.")

        self._sqrt_embedding_dim = math.sqrt(self.embedding_net.output_size)

        # positional encoder
        self._positional_encoding_type = cfg.transformer_positional_encoding_type
        if self._positional_encoding_type.lower() == 'concatenate':
            encoder_dim = self.embedding_net.output_size * 2
        elif self._positional_encoding_type.lower() == 'sum':
            encoder_dim = self.embedding_net.output_size
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self._positional_encoding_type}")
        if cfg.timeF==False:
            self.positional_encoder = _PositionalEncoding(embedding_dim=self.embedding_net.output_size,
                                                      dropout=cfg.transformer_positional_dropout,
                                                      position_type=cfg.transformer_positional_encoding_type,
                                                      max_len=cfg.seq_length)
        else:
            self.positional_encoder = DataEmbedding(128, 128, 'timeF', 'd', 0.1)

        # positional mask
        self._mask = None

        # encoder
        self.encoder = encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, encoder_dim, n_heads, bucket_size=bucket_size,
                                  n_hashes=n_hashes),
                    encoder_dim,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(encoder_dim)
        )

        # head (instead of a decoder)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=encoder_dim, n_out=self.output_size)
        self.timeF=cfg.timeF


    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on a transformer model without decoder.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
        """
        # pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)
        device = data['x_t'].device

        if self.timeF==True:
            x_t = pd.to_datetime(data['x_t'].cpu().data.numpy().flatten().astype(str), format='%Y%m%d')#########
            x_t = time_features(x_t, freq='d').transpose(1, 0).reshape(data['x_t'].shape[0], data['x_t'].shape[1], -1)############
            x_t = x_t.astype(np.float32)
            x_t = torch.from_numpy(x_t).to(device)
            positional_encoding = self.positional_encoder(x_d.transpose(0, 1), x_t)###########
            
        else:########3
            positional_encoding = self.positional_encoder(x_d * self._sqrt_embedding_dim).transpose(0, 1) ######### #after embedding [batch_size, seq_len, d_model]


        # mask out future values
        if self._mask is None or self._mask.size(0) != len(x_d):
            self._mask = torch.triu(x_d.new_full((len(x_d), len(x_d)), fill_value=float('-inf')), diagonal=1)

        # encoding
        output, attns = self.encoder(positional_encoding, )

        # head
        pred = self.head(self.dropout(output))

        # add embedding and positional encoding to output
        pred['embedding'] = x_d
        pred['positional_encoding'] = positional_encoding

        return pred


class _PositionalEncoding(nn.Module):
    """Class to create a positional encoding vector for timeseries inputs to a model without an explicit time dimension.

    This class implements a sin/cos type embedding vector with a specified maximum length. Adapted from the PyTorch
    example here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Parameters
    ----------
    embedding_dim : int
        Dimension of the model input, which is typically output of an embedding layer.
    dropout : float
        Dropout rate [0, 1) applied to the embedding vector.
    max_len : int, optional
        Maximum length of positional encoding. This must be larger than the largest sequence length in the sample.
    """

    def __init__(self, embedding_dim, position_type, dropout, max_len=5000):
        super(_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, int(np.ceil(embedding_dim / 2) * 2))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(max_len * 2) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :embedding_dim].unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        if position_type.lower() == 'concatenate':
            self._concatenate = True
        elif position_type.lower() == 'sum':
            self._concatenate = False
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {position_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for positional encoding. Either concatenates or adds positional encoding to encoder input data.

        Parameters
        ----------
        x : torch.Tensor
            Dimension is ``[sequence length, batch size, embedding output dimension]``.
            Data that is to be the input to a transformer encoder after including positional encoding.
            Typically this will be output from an embedding layer.

        Returns
        -------
        torch.Tensor
            Dimension is ``[sequence length, batch size, encoder input dimension]``.
            The encoder input dimension is either equal to the embedding output dimension (if ``position_type == sum``)
            or twice the embedding output dimension (if ``position_type == concatenate``).
            Encoder input augmented with positional encoding.

        """
        if self._concatenate:
            x = torch.cat((x, self.pe[:x.size(0), :].repeat(1, x.size(1), 1)), 2)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

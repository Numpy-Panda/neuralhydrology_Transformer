import logging
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('neuralhydrology/modelzoo/FEDformer/')

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from reformer_pytorch import LSHSelfAttention
import torch.nn.functional as F
import pandas as pd
from neuralhydrology.modelzoo.time_features_embedding import time_features 
from neuralhydrology.modelzoo.FEDformer.models.FEDformer import Model



LOGGER = logging.getLogger(__name__)


class Fedformer_Configs(object):
    def __init__(self, cfg, dim):
        self.modes = cfg.fedformer_modes#32
        self.mode_select = cfg.fedformer_mode_select#'random'#
        self.version = cfg.fedformer_version#'Wavelets'#

        self.moving_avg = [12, 24]
        self.L = 1
        self.base = cfg.fedformer_base# 'legendre'  ##'chebyshev'#
        self.cross_activation = 'tanh'
        self.seq_len = cfg.seq_length
        self.label_len = 0
        self.pred_len = cfg.predict_last_n
        self.output_attention = True
        self.enc_in = dim
        self.dec_in = dim
        self.d_model = 16
        self.embed = 'timeF'
        self.dropout = 0.05
        self.freq = 'd'
        self.factor = cfg.fedformer_factor#1
        self.n_heads = cfg.fedformer_nheads#8
        self.d_ff = 16
        self.e_layers = cfg.fedformer_e_layers#2
        self.d_layers = cfg.fedformer_d_layers#1
        self.c_out = dim
        self.activation = 'gelu'
        self.wavelet = 0
    


class FEDformer(BaseModel):
    """FEDformer model class.
    
    The model configuration is specified in the config file using the following options:

    * ``fedformer_version``: two subversionstructures for signal process, can be 'Wavelets' or 'Fourier'.
    * ``fedformer_base``: wavelet orthogonal polynomials, can be 'legendre' or 'chebyshev'.
    * ``fedformer_mode_select``: mode select method, can be 'random' or else.
    * ``fedformer_nheads``: number of heads.
    * ``fedformer_e_layers``: number of encoder layers.
    * ``fedformer_d_layers``: number of decoder layers.
    * ``fedformer_factor``: Probsparse attn factor.
    * ``fedformer_modes``: Number of modes (defaults to 32).

    Parameters
    ----------
    cfg : Config
        The run configuration.
        
    References
    ----------
    .. [#] Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency 
    enhanced decomposed transformer for long-term series forecasting. arXiv preprint arXiv:2201.12740.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super(FEDformer, self).__init__(cfg=cfg)
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        
       

        # embedding net before transformer
        self.embedding_net = InputLayer(cfg)

        # ensure that the number of inputs into the self-attention layer is divisible by the number of heads
        self._positional_encoding_type = cfg.transformer_positional_encoding_type
        if self._positional_encoding_type.lower() == 'concatenate':
            encoder_dim = self.embedding_net.output_size * 2
        elif self._positional_encoding_type.lower() == 'sum':
            encoder_dim = self.embedding_net.output_size
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self._positional_encoding_type}")


            
        # define the model
        self.head = get_head(cfg=cfg, n_in=encoder_dim, n_out=self.output_size)
        self.fedformer_config = Fedformer_Configs(cfg, self.embedding_net.output_size)
        self.model = Model(self.fedformer_config)
        print('********************')
        print('You are using FEDformer')
        print(f'{cfg.fedformer_version} for signal process')
        print(f'Wavelet orthogonal polynomials: {cfg.fedformer_base}')
        print(f'Mode select method: {cfg.fedformer_mode_select}')
        print(f'Number of attention heads: {cfg.fedformer_nheads}')
        print(f'Number of FEDformer encoder layers: {cfg.fedformer_e_layers}')
        print(f'Number of FEDformer decoder layers: {cfg.fedformer_d_layers}')
        print(f'Probsparse attn factors: {cfg.fedformer_factor}')
        print(f'Number of modes: {cfg.fedformer_modes}')
        print('********************')


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
        x_d = self.embedding_net(data)  #[seq_len, batch_size, d_model]
        device = x_d.device

        x_enc_mark = pd.to_datetime(data['x_t'].numpy().flatten().astype(str), format='%Y%m%d')
        x_enc_mark = time_features(x_enc_mark, freq='d').transpose(1, 0).reshape(data['x_t'].shape[0], data['x_t'].shape[1], -1)
        last_day = pd.to_datetime(data['x_t'][:, -1].numpy().flatten().astype(str), format='%Y%m%d')
        next_day = pd.DatetimeIndex(last_day) + pd.DateOffset(1)
        next_day = time_features(next_day, freq='d').transpose(1, 0).reshape(data['x_t'].shape[0], 3)
        x_dec_mark = np.insert(x_enc_mark, data['x_t'].shape[1], values=next_day, axis=1)
            
        x_enc_mark = x_enc_mark.astype(np.float32)
        x_enc_mark = torch.from_numpy(x_enc_mark)
            
        x_dec = x_d
        x_dec_mark = x_dec_mark.astype(np.float32)
        x_dec_mark = torch.from_numpy(x_dec_mark)
            
            
        x_enc = x_d.permute(1, 0, 2).to(device)
        x_enc_mark = x_enc_mark.to(device)
        x_dec = x_dec.to(device)
        x_dec_mark = x_dec_mark.to(device)

        # encoding
        #output, attns = self.encoder(positional_encoding, )  #output [batch_size, seq_len, d_model]
        output, _ = self.model(x_enc, x_enc_mark, x_dec, x_dec_mark)

        

        # head
        pred = self.head(self.dropout(output))

        # add embedding and positional encoding to output
        pred['embedding'] = x_d
        #pred['positional_encoding'] = positional_encoding
        


        return pred


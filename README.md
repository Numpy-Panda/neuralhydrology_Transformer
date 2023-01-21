# Transformer-based rainfall-runoff modeling

Over the past several months, I have undertaken research on the application of the [Transformer](https://arxiv.org/abs/1706.03762) architecture to rainfall-runoff modeling. In recent years, the availability of large-sample hydrological datasets (e.g. [CAMELS](https://ral.ucar.edu/solutions/products/camels)) and open-source code (e.g. [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)) has facilitated the rapid advancement of machine learning in the field of hydrology. In this light, I am publicly releasing the code, which constitutes a crucial component of my Master's thesis (available in [TU Delft repository](https://repository.tudelft.nl/) soon). Building upon the NeuralHydrology framework, the repository incorporates several rainfall-runoff models based on variants of the Transformer architecture, including the [Reformer](https://arxiv.org/abs/2001.04451), [Informer](https://arxiv.org/abs/2012.07436), [Linformer](https://arxiv.org/abs/2006.04768), and [FEDformer](https://arxiv.org/abs/2201.12740), which have been specifically developed for time-series prediction based on Transformer architecture.




When applying deep learning technique for rainfall-runoff modeling, the discharge at a particular time step is a function of the meteorological forcing observed over the past n time steps. Therefore, compared to the original Transformer architecture, the Transformer used for hydrological modeling only needs the Encoder part.

![#](docs/source/_static/img/Transformers_for_RR.svg)

These Transformer variants were used for regional rainfall-runoff modeling using the CAMELS dataset, and their performance was compared to a benchmark LSTM. The results can be observed in the metrics presented below.

|           | Reformer | FEDformer | Linformer | Transformer | Informer |    LSTM |
|-----------|---------:|----------:|----------:|------------:|---------:|--------:|
| NSE       |    0.728 |     0.732 |     0.715 |       0.725 |    0.709 |   0.732 |
| Alpha-NSE |    0.822 |     0.858 |     0.866 |       0.834 |    0.817 |   0.842 |
| Beta-NSE  |   -0.050 |    -0.011 |    -0.016 |      -0.006 |   -0.032 |  -0.039 |
| FHV       |  -17.376 |   -14.263 |   -13.368 |     -15.900 |  -17.717 | -16.058 |
| FLV       |  -19.797 |    -1.806 |  -177.769 |      24.667 |    3.661 |  28.927 |
| FMS       |    3.385 |    -9.191 |   -11.540 |     -13.742 |   -7.671 |  -8.021 |


# Usage


Some Transformer variants has been added into the NeuralHydrology.

## Reformer
The detailed descriptions about the arguments of Reformer are as following:

| Hyperparameter name | Description of parameter |
| --- | --- |
| reformer_layers           | Number of reformer encoder layers (defaults to 2).                           |
| reformer_nheads      | Number of attention heads (defaults to 2).    |
| reformer_bucket_size      | Nash bucket size (defaults to 16).                  |
| reformer_n_hashes      |  Number of hash (defaults to 4).               |
| reformer_dropout      | Reformer dropout (defaults to 0.1).  |


## Informer
The detailed descriptions about the arguments of Informer are as following:
| Hyperparameter name | Description of parameter |
| --- | --- |
| informer_n_layers           | Number of encoder layers (defaults to 2).       |
| informer_n_heads      | Number of heads (defaults to 4).  |
| informer_distil      | Whether to use distilling in encoder, using this argument means not using distilling (defaults to `True`).                  |
| informer_factor      | Probsparse attn factor (defaults to 5).             |
| informer_activation      | Activation function (defaults to `gelu`).  |

## FEDformer
The detailed descriptions about the arguments of FEDformer are as following:
| Hyperparameter name | Description of parameter |
| --- | --- |
| fedformer_version           | Two subversionstructures for signal process, can be `Wavelets` or `Fourier`. |
| fedformer_base      | Wavelet orthogonal polynomials, can be `legendre` or `chebyshev`. |
| fedformer_mode_select      | Mode select method, can be `random` or `not`.  |
| fedformer_nheads      | Number of attention heads (defaults to 8).  |
| fedformer_e_layers      | Number of FEDformer encoder layers (defaults to 2).    |
| fedformer_d_layers      | Number of FEDformer decoder layers (defaults to 1).  |
| fedformer_factor      | Probsparse attn factor (defaults to 1).  |
| fedformer_modes      | Number of modes (defaults to 32).  |






## Linformer
The detailed descriptions about the arguments of FEDformer are as following:
| Hyperparameter name | Description of parameter |
| --- | --- |
| linformer_n_layers           | Number of Linformer encoder layers (defaults to 2).                                             |
| linformer_n_heads      | number of Linformer heads (defaults to 2).    |



# Cite

In case you use any Transformer models in your research or work, it would be highly appreciated if you cite this repository.

If this repository is helpful for you, please give me a star!

# Contact


In special cases, you can also reach out to me by email: maokangmin123(at)gmail.com, or comment at my personal website: [kangmin.nl](http://kangmin.nl/).


# Recommendation

According to our results from the [CAMELS](https://ral.ucar.edu/solutions/products/camels) experiment, the Reformer model with time feature positional encoding performed well in simulating the rainfall-runoff relationship in snow-driven basins (major precipitation form is snow), even outperforming LSTM. Therefore, if you are looking to do hydrological modeling in a snow-driven catchment, the Reformer model may be worth considering.

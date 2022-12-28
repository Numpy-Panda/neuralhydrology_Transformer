![#](docs/source/_static/img/neural-hyd-logo-black.png)

This repository is based on [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) and has incorporated several [Transformer](https://arxiv.org/abs/1706.03762) models and [Timestamp Positional Encoding method](https://arxiv.org/abs/2012.07436). The Transformer models include [Reformer](https://arxiv.org/abs/2001.04451), [Informer](https://arxiv.org/abs/2012.07436), [Linformer](https://arxiv.org/abs/2006.04768), and [FEDformer](https://arxiv.org/abs/2201.12740), which were introduced in recent years to address the issues with the original Transformer in time series prediction problems. 

[]()

# Transformers for Hydrological modeling
In a deep learning-based hydrological model, the discharge at a particular time step is a function of the meteorological forcing observed over the past n time steps. Therefore, compared to the original Transformer architecture, the Transformer used for hydrological modeling only needs the Encoder part.

![#](docs/source/_static/img/Transformers_for_RR.svg)




# Cite

In case you use any Transformer models in your research or work, it would be highly appreciated if you cite this repository.

# Contact

For questions or comments regarding the usage of this repository, please use the [discussion section](https://github.com/neuralhydrology/neuralhydrology/discussions) on Github. For bug reports and feature requests, please open an [issue](https://github.com/neuralhydrology/neuralhydrology/issues) on GitHub.
In special cases, you can also reach out to us by email: neuralhydrology(at)googlegroups.com


# Recommendation

According to our results from the [CAMELS](https://ral.ucar.edu/solutions/products/camels) experiment, the Reformer model with time feature positional encoding performed well in simulating the rainfall-runoff relationship in snow-driven basins (major precipitation form is snow), even outperforming LSTM. Therefore, if you are looking to do hydrological modeling in a snow-driven catchment, the Reformer model may be worth considering.

# VSN-SparseTransformer
This is a combination of the variable selection network from [Temporal Fusion Transformers for interpretable multi-horizon time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207021000637#b15)
and the sparse attention layer from [Temporal Convolutional Attention Neural Networks for Time Series Forecasting](https://ieeexplore.ieee.org/abstract/document/9534351) for time series classification.
The idea behind this is that the variable selection network will negate the influence of uninformative features while the sparse attention network will select the most 
informative time steps. Since these are computationally expensive, the model uses a cyclical learning rate similar to [fastai's one cycle policy](https://www.fast.ai/2018/07/02/adam-weight-decay/) to achieve superconvergence in few epochs.

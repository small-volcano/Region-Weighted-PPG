# Training protocol

The primary model uses convolution channels 32, 64, and 128, kernels 9, 7, and 5, stride 2, ReLU activations, adaptive average pooling, a 128-dimensional projection, L2-normalized representations, and a two-output linear head. The two standardized outputs are SBP and DBP.

Per-example Smooth-L1 losses use PyTorch's default beta of 1 and are averaged across outputs. ERM uses equal sample weights. The SBP, DBP, and union objectives weight the corresponding training-label region with the configured multiplier. In each batch the weighted loss is normalized by the sum of weights.

The recorded primary configuration trains for five epochs using AdamW, learning rate 0.001, weight decay 0.0001, and batch size 1024. Seeds 42, 43, and 44 were used in the original primary comparison; additional runs used 45 and 46. The final training epoch is saved. SBP and DBP are standardized using training-set means and population standard deviations; PPG windows use independent mean and population-standard-deviation normalization with epsilon 1e-6.

The source MAT file uses MATLAB v7.3 / HDF5 arrays with signals shaped `(time, channels, windows)` and labels shaped `(1, windows)`. Waveforms and labels must be finite. This release uses the full training subset provided by the user. The output checkpoint records the configuration, training-window count, and target normalization values.

The subject-held-out validation selection and alternative-objective experiments use separate training protocols. This release focuses on the primary training objective and architecture.

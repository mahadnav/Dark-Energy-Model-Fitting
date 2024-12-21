This code is provided for one of the cosmological models, namely the CPL Model, forming a computational understanding behind model implementation, likelihood estimation and AIC/BIC calculation for parameter optimization.

### Model Computation for Parameter Selection

`numba` version 0.59.0 is used for JIT compilation.

The script loads data from a CSV file (`Hz_2016.csv`) containing redshifts (`z`), Hubble parameter values (`Hz`), and errors (`sig`).

The results are saved as a CSV file with chi-squared, likelihood, normalized chi-squared, AIC, BIC, and parameter values.

### Plotting

`chainconsumer` version 0.34.0 is used for plotting the results.

The plotting script visualizes the results of the CPL Cosmological Parameter Estimation by creating a contour plot using the ChainConsumer library. The contour plot includes parameters such as Omega Lambda, H0, w0, and wa (depending on the model you're trying to evaluate).

This code is provided for one of the cosmological models, namely the CPL Model, forming a computational understanding behind model implementation, likelihood estimation and AIC/BIC calculation for parameter optimization.



### Import necessary libraries:

```python
from __future__ import print_function
import threading
import numba as nb
import numpy as np
from numpy import genfromtxt
import datetime
import mpmath as smp
```

These lines import the required libraries: `numba` for JIT compilation, `numpy` for numerical operations, and `mpmath` for arbitrary-precision arithmetic.

### Load data from a CSV file:

```python
szdat = np.genfromtxt(r"C:\Users\ATOnline\Desktop\Hz_2016.csv", delimiter=',', skip_header=0)
z = szdat[:, 0]
Hz = szdat[:, 1]
sig = szdat[:, 2]
Hz_error2 = sig**2
```

The script loads data from a CSV file (`Hz_2016.csv`) containing redshifts (`z`), Hubble parameter values (`Hz`), and errors (`sig`). The error squared (`Hz_error2`) is calculated.

### Define constants:

```python
c = 299792.45
Simpson_N = 16
```

These lines define the speed of light (`c`) and a constant for numerical integration (`Simpson_N`).

### Define functions to calculate AIC and BIC:

```python
def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic

def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic
```

These functions calculate the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

### Define a function for the CPL model:

```python
@nb.jit('float64(float64, float64, float64, float64, float64)')
def Hz_CPL(zh, Om_l, H0, w0, wa):
    Omega_r = 0
    Omega_m = 1 - Om_l
    Omega_lambda = Om_l 
    z = zh
    E = np.sqrt((Omega_lambda)*np.exp(-(3*wa*z)/(1+z))*(1 + z)**(3*(1 + wa + w0)) + (Omega_m*((1+z)**3)) + (Omega_r*(1+z)**4))
    return float(H0 * (E))
```

This function defines the CPL (Chevallier-Polarski-Linder) model for the Hubble parameter as a function of redshift.

### Set up parameters for grid search:

```python
n = 50
param = 4
ols = np.linspace(0.6, .9, n)
H0s = np.linspace(60., 90., n)
w0s = np.linspace(-5, 0, n)
was = np.linspace(-1, 3, n)
```

These lines set up the parameters for the grid search, defining ranges for `Omega_lambda`, `H0`, `w0`, and `wa`.

### Define a function to calculate chi-squared values and other parameters for all parameter combinations:

```python
@nb.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])')
def Chisq_list(ols, H0s, chi2, olv, H0v, w0_vec, wa_vec, aic_values, bic_values):
    x = 0
    for i in range(0, len(ols)):
        for j in range(0, len(H0s)):
            for k in range(0, len(w0s)):
                for l in range(len(was)):
                    Hz_model = Hz_CPL_v(z, ols[i], H0s[j], w0s[k], was[l])
                    chi2_test = np.sum((Hz_model - Hz)**2 / Hz_error2)
                    # ... (other parameter values and calculations)
                    x += 1
```

This function performs a nested loop over the parameter space, calculating chi-squared values, parameter values, and other relevant quantities for each combination.

### Run the grid search:

```python
Chisq_list(ols, H0s, chi2, olv, H0v, w0_vec, wa_vec, aic_values, bic_values)
```

This line calls the function to perform the grid search and populate arrays with chi-squared values and corresponding parameter values.

### Post-process the results:

```python
likelihood = np.exp(-0.5 * (chi2 - chi2.min()))
chi_sq_dof = chi2 / (len(z) - 1 - param)
print('Chi square min: ', chi2.min())
print('AIC min: ', aic_values.min())
print('BIC min: ', bic_values.min())
```

These lines calculate likelihood and normalized chi-squared values, and print minimum chi-squared, AIC, and BIC values.

### Save results to a CSV file:

```python
dir = r"C:\Users\ATOnline\Desktop\CPL_Chisq.csv"
with open(dir, "w") as file:
    header = "Chi2, Likelihood, Chi_sq_DOF, AIC, BIC, Omega Lambda, H0, w0, wa\n"
    file.write(header)
    chisq_file = np.column_stack((chi2, likelihood, chi_sq_dof, aic_values, bic_values, olv, H0v, w0_vec, wa_vec))
    np.savetxt(file, chisq_file, delimiter=",")
```

This block writes the results to a CSV file with headers and columns for chi-squared, likelihood, normalized chi-squared, AIC, BIC, and parameter values.

### Visualization

The plotting script visualizes the results of the CPL Cosmological Parameter Estimation by creating a corner plot using the ChainConsumer library. The corner plot includes parameters such as Omega Lambda, H0, w0, and wa (depending on the model you're trying to evaluate).

from __future__ import print_function
import threading
import numba as nb
import numpy as np
from numpy import genfromtxt
import datetime
from numba import float64, guvectorize
import mpmath as smp

Simpson_N = 16

szdat = np.genfromtxt(r"C:\Users\ATOnline\Desktop\Hz_2016.csv", 
                        delimiter=',', skip_header=0)
z = szdat[:, 0]
Hz = szdat[:, 1]
sig = szdat[:, 2]
Hz_error2 = sig**2

c = 299792.45

# Function to calculate AIC
def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic

# Function to calculate BIC
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic

@nb.jit('float64(float64, float64, float64, float64, float64)')
def Hz_CPL(zh, Om_l, H0, w0, wa):
        Omega_r = 0
        Omega_m = 1 - Om_l
        Omega_lambda = Om_l 
        z = zh
        E = np.sqrt((Omega_lambda)*np.exp(-(3*wa*z)/(1+z))*(1 + z)**(3*(1 + wa + w0)) + (Omega_m*((1+z)**3)) + (Omega_r*(1+z)**4))
        return float(H0 * (E))

Hz_CPL_v = np.vectorize(Hz_CPL)
n = 50
param = 4

# Parameters
ols = np.linspace(0.6, .9, n)
H0s = np.linspace(60., 90.,n)
w0s = np.linspace(-5, 0, n)
was = np.linspace(-1, 3, n)

# Vectorized params
chi2 = np.ones(n**param) * np.infty
olv = np.ones(n**param) * np.infty
H0v = np.ones(n**param) * np.infty
w0_vec = np.ones(n**param) * np.infty
wa_vec = np.ones(n**param) * np.infty
aic_values = np.ones(n**param) * np.infty
bic_values = np.ones(n**param) * np.infty

@nb.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])')
def Chisq_list(ols, H0s, chi2, olv, H0v, w0_vec, wa_vec, aic_values, bic_values):
    x=0
    for i in range(0, len(ols)):
            for j in range(0, len(H0s)):
                    for k in range(0, len(w0s)):
                            for l in range(len(was)):
                                Hz_model = Hz_CPL_v(z, ols[i], H0s[j], w0s[k], was[l])
                                chi2_test = np.sum((Hz_model - Hz)**2 / Hz_error2)
                                chi2[x] = chi2_test
                                olv[x] = ols[i]
                                H0v[x] = H0s[j]
                                w0_vec[x] = w0s[k]
                                wa_vec[x] = was[l]

                                # Calculate AIC and BIC
                                aic_values[x] = calculate_aic(len(z), chi2_test, param)
                                bic_values[x] = calculate_bic(len(z), chi2_test, param)

                                x += 1

Chisq_list(ols, H0s, chi2, olv, H0v, w0_vec, wa_vec, aic_values, bic_values)
likelihood = np.exp(-0.5 * (chi2 - chi2.min()))
chi_sq_dof = chi2 / (len(z) - 1 - param)
print('Chi square min: ', chi2.min())
print('AIC min: ', aic_values.min())
print('BIC min: ', bic_values.min())

dir = r"C:\Users\ATOnline\Desktop\CPL_Chisq.csv"
# Create a file and add a header
with open(dir, "w") as file:
    # Write the header
    header = "Chi2, Likelihood, Chi_sq_DOF, AIC, BIC, Omega Lambda, H0, w0, wa\n"
    file.write(header)

    # Create a numpy array with your data
    chisq_file = np.column_stack((chi2, likelihood, chi_sq_dof, aic_values, bic_values, olv, H0v, w0_vec, wa_vec))

    # Write the data
    np.savetxt(file, chisq_file, delimiter=",")

print("\nComputation complete!")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from chainconsumer import ChainConsumer

params = {'text.usetex':True,'font.size':1,'font.family':
          'serif','figure.autolayout': False}
plt.rcParams.update(params)

szdat=np.genfromtxt(r"C:\Users\ATOnline\Desktop\CPL_Chisq.csv", delimiter=",", skip_header=True)

likelihood = szdat[:, 1]
olv = szdat[:, 5]
H0v = szdat[:, 6]
w0 = szdat[:, 7]
wa = szdat[:, 8]

c = ChainConsumer()
c.add_chain([olv, H0v, w0, wa], weights=likelihood, grid=True, 
            parameters=[r"$\Omega_\Lambda$", r"$H_0$", r"$w_0$", r"$w_a$"]).configure(statistics='mean')
c.plotter.plot()
plt.show()
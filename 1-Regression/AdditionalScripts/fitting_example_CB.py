import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#############################################################################################################################
# Load data

fontsize = 16
fileName = "../Histogram/mass_mm_higgs_Signal.npz"

with np.load(fileName) as data:
    bin_edges = data['bin_edges']
    bin_centers = data['bin_centers']
    bin_values = data['bin_values']
    bin_errors = data['bin_errors']


#############################################################################################################################
# Crystal Ball Function

def CrystalBall(x, A, aL, aR, nL, nR, mCB, sCB):
    return np.piecewise(x, [(x - mCB) / sCB <= -aL,
                            (x - mCB) / sCB >= aR],
        [lambda x: A * (nL / np.abs(aL))**nL * np.exp(-aL**2 / 2) * (nL / np.abs(aL) - np.abs(aL) - (x - mCB) / sCB)**(-nL),
         lambda x: A * (nR / np.abs(aR))**nR * np.exp(-aR**2 / 2) * (nR / np.abs(aR) - np.abs(aR) + (x - mCB) / sCB)**(-nR),
         lambda x: A * np.exp(-(x - mCB)**2 / (2 * sCB**2))
    ])

#############################################################################################################################
# Fit

popt, pcov = curve_fit(CrystalBall, bin_centers, bin_values, sigma=bin_errors, p0=[133., 1.5, 1.5, 3.7, 9.6, 124.5, 3.])
perr = np.sqrt(np.diag(pcov))
A, aL, aR, nL, nR, mCB, sCB = popt

xs = np.linspace(110, 160, 501)
my_fit = np.array(CrystalBall(xs, A, aL, aR, nL, nR, mCB, sCB))
xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])

plt.figure(figsize=(8, 4.5))
plt.errorbar(bin_centers, bin_values, bin_errors, xerrs, marker='o', markersize=5, color='k', ecolor='k', ls='', label='Original histogram')
plt.plot(xs, my_fit, 'r-', label='signal fit CB')
plt.xlabel(r'$m_{\mu \mu}$', fontsize=fontsize)
plt.ylabel('Number of events', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.xticks(size=fontsize)
plt.yticks(size=fontsize)
plt.tight_layout()
plt.grid()
plt.show()

# Print out the parameters
print("A: {A:.3f}\naL: {aL:.3f}\naR: {aR:.3f}\nnL: {nL:.3f}\nnR: {nR:.3f}\nmCB: {mCB:.3f}\nsCB: {sCB:.3f}".format(A=A, aL=aL, aR=aR, nL=nL, nR=nR, mCB=mCB, sCB=sCB))

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from atlas_fit_function import atlas_invMass_mumu_core

# Load data:
inFileName = "../Histogram/mass_mm_higgs_Background.npz"
with np.load(inFileName) as data:
    bin_edges = data['bin_edges']
    bin_centers = data['bin_centers']
    bin_values = data['bin_values']
    bin_errors = data['bin_errors']

#############################################################################################################################


def poly3(x, a, b, c, d):
    """Fit poly3"""
    return a * x**3 + b * x**2 + c * x + d

def poly4(x, a0, a, b, c, d):
    """Fit poly4"""
    return a0*x**4 + a * x**3 + b * x**2 + c * x + d


fitfun = poly4
popt, pcov = curve_fit(fitfun, bin_centers, bin_values, sigma=bin_errors, p0=[1,1, 1, 1, 1])
perr = np.sqrt(np.diag(pcov))
a0, a, b, c, d = popt

my_fit = np.array(fitfun(bin_centers, a0, a, b, c, d))

xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])

plt.figure()
plt.errorbar(bin_centers, bin_values, bin_errors, xerrs, fmt="none", color='b', ecolor='b', label='Original histogram')
plt.plot(bin_centers, my_fit, 'g-', label='fit poly4')
plt.legend()
plt.ylabel("Num. of events / bin")
plt.xlabel(r"Energy $m_{\mu \mu}$ [GeV]")
plt.savefig("../Pictures/bkg_polyfit4.pdf")
plt.show()

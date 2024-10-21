import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from atlas_fit_function import atlas_invMass_mumu_core
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin


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



class VandermondeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vander(X, N=self.degree)

alpha=0.01
M = 10 #monomial order
pipeline = Pipeline([
    ("expand", VandermondeFeatures(M)),
    ("normalize",StandardScaler()),
    ("ridge",Ridge(alpha=alpha))])
model = TransformedTargetRegressor(regressor=pipeline, transformer = StandardScaler())


Y = bin_values
X = bin_centers
model.fit(X,Y)
my_fit = model.predict(X)

xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])

plt.figure()
plt.errorbar(bin_centers, bin_values, bin_errors, xerrs, fmt="none", color='b', ecolor='b', label='Original histogram')
plt.plot(bin_centers, my_fit, 'g-', label=f'fit poly10, alpha=0.01')
plt.legend()
plt.ylabel("Num. of events / bin")
plt.xlabel(r"Energy $m_{\mu \mu}$ [GeV]")
plt.savefig("../Pictures/bkg_ridge3.pdf")
plt.show()

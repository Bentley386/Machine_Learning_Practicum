import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from atlas_fit_function import atlas_invMass_mumu_core
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

# Load data:
inFileName = "../Histogram/mass_mm_higgs_Background.npz"
with np.load(inFileName) as data:
    bin_edges = data['bin_edges']
    bin_centers = data['bin_centers']
    bin_values = data['bin_values']
    bin_errors = data['bin_errors']

#############################################################################################################################

C=10
epsilon=0.005

pipeline = Pipeline([
    ("normalize",StandardScaler()),
    ("svr",SVR(kernel="poly",C=C,epsilon=epsilon))])

model = TransformedTargetRegressor(regressor=pipeline, transformer = StandardScaler())

X = bin_centers.reshape(-1,1)
Y = bin_values
print(X.shape)
print(Y.shape)
model.fit(X,bin_values)
my_fit = model.predict(X)

xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])

plt.figure()
plt.errorbar(bin_centers, bin_values, bin_errors, xerrs, fmt="none", color='b', ecolor='b', label='Original histogram')
plt.plot(bin_centers, my_fit, 'g-', label=f'poly3  kernel, C={C}, eps={epsilon}')
plt.legend()
plt.ylabel("Num. of events / bin")
plt.xlabel(r"Energy $m_{\mu \mu}$ [GeV]")
plt.savefig("../Pictures/bkg_svr0.pdf")
plt.show()

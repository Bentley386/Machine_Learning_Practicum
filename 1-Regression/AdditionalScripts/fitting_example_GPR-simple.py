import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
# Different kernels available in sklearn:
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, ExpSineSquared, RationalQuadratic
from matplotlib import pyplot as plt
# Set random seed for same results
np.random.seed(12345)

# Load data
inFileName = "../Histogram/mass_mm_higgs_Background.npz"
with np.load(inFileName) as data:
    bin_edges = data['bin_edges']
    bin_centers = data['bin_centers']
    bin_values = data['bin_values']
    bin_errors = data['bin_errors']

# Set hyper-parameter bounds for ConstantKernel
nEvts = np.sum(bin_values)
const0 = 1
const_low = 10.0**-5
const_hi = nEvts * 10.0

# Set hyper-parameter bounds for RBF kernel
RBF0 = 1
RBF_low = 1e-2
RBF_high = 1e2

# A) Define kernel: ConstantKernel * RBF:
kernel_RBF = ConstantKernel(const0, constant_value_bounds=(const_low, const_hi)) * RBF(RBF0, length_scale_bounds=(RBF_low, RBF_high))

# B) Define kernel: ConstantKernel * Matern:
kernel_Matern = ConstantKernel(const0, constant_value_bounds=(const_low, const_hi)) * Matern(RBF0, length_scale_bounds=(RBF_low, RBF_high), nu=1.5)

# Transform x data into 2d vector!
X = np.atleast_2d(bin_centers).T  # true datapoints
X_to_predict = np.atleast_2d(np.linspace(110, 160, 1000)).T  # what to predict
y = bin_values

# Initialize Gaussian Process Regressor: !!! alpha = 2xbin_errors or 1xbin_errors ? Your task to figure out. !!!
gp = GaussianProcessRegressor(kernel=kernel_RBF, n_restarts_optimizer=1, alpha=2 * bin_errors)

# Fit on X with values y
gp.fit(X, y)

# Predict
y_pred, sigma = gp.predict(X_to_predict, return_std=True)

plt.title("Example GPR with RBF kernel", fontsize=14, fontweight='bold')
plt.xlabel(r"$m_{\mu\mu}$ [GeV]", fontsize=12)
plt.ylabel("Events/Bin", fontsize=12)
plt.scatter(bin_centers, bin_values, color='r', linewidth=0.5, marker='o', s=10)
plt.plot(X_to_predict, y_pred, color='k')
plt.fill_between(X_to_predict.ravel(), y_pred - sigma, y_pred + sigma)
plt.savefig("../Pictures/GPR_simple.pdf")
plt.show()

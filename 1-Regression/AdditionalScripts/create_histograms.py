import os
import sys
import numpy as np
import pandas as pd

# User defined: -> play around with n_bins and x_range
#################################################################################
datadir_input = "../download/"  # Directory to raw data, change this!
datadir_output = "../Histogram/"  # Directory to generated histograms, change/create this!
x_range = (110, 160)  # m_mumu energy interval (110.,160.) for Higgs
n_bins = 250  # number of bins for histograms
#################################################################################

ds_bkg = "mc_bkg_new"  # filename for Background simulations (.h5)
ds_sig = "mc_sig"  # filename for Signal simulations (.h5)
ds_data = "data"  # filename for Measured data (.h5)

datasets = [ds_bkg, ds_sig, ds_data]
labels = ["Background", "Signal", "Data"]


def load_data(dataset, datadir):
    """Function for loading .h5 Atlas datasets"""
    infile = os.path.join(datadir, dataset + '.h5')
    print('Loading {}...'.format(infile))
    store = pd.HDFStore(infile, 'r')
    dataset = store['ntuple']

    return dataset, store


for label, dataset in zip(labels, datasets):
    # Load dataset
    ds, file = load_data(dataset, datadir_input)

    # Get simulated (Background, Signal) or measured (Data) data
    all_events = ds["Muons_Minv_MuMu_Paper"]

    # Get correct weights
    wts = ds["CombWeight"]
    wts2 = wts * wts

    # Firstly, get correct number of bin_values
    bin_values, _ = np.histogram(all_events, bins=n_bins, range=x_range, weights=wts)  # wts!

    # Secondly, calculate bin_errors
    y, bin_edges = np.histogram(all_events, bins=n_bins, range=x_range, weights=wts2)  # wts2!
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_errors = np.sqrt(y)

    # Finally, save several arrays into a single file in uncompressed .npz format
    save_name = datadir_output + 'hist_range_' + str(x_range[0]) + '-' + str(x_range[1]) + '_nbin-' + str(n_bins) + '_' + label + '.npz'
    with open(save_name, 'wb') as f:
        np.savez(f, bin_edges=bin_edges, bin_centers=bin_centers, bin_values=bin_values, bin_errors=bin_errors)
    f.close()

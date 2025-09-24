import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot data from HDF5 files.")
parser.add_argument('filenames', nargs='+', help='List of HDF5 filenames to plot')
args = parser.parse_args()

for filename in args.filenames:
    with h5py.File(filename, 'r') as file:
        mass = file['state']['mass'][:]
        r0, r1, dlogr = file['config']['domain'][()]

    rf = np.logspace(np.log10(r0), np.log10(r1), len(mass) + 1)
    rc = 0.5 * (rf[:-1] + rf[1:])
    dA = np.pi * np.diff(rf**2)
    sigma = mass / dA

    plt.plot(rc, sigma, label=filename)

plt.xlabel(r'$r$')
plt.ylabel(r'$\Sigma$')
plt.grid(True)
plt.legend()
plt.show()

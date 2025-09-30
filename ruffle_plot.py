import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse


def ring_sigma(m, r, t, r0, nu):
    from scipy.special import iv
    tau = 12.0 * nu * t / (r0 * r0)
    x = r / r0
    return m / (np.pi * r0**2) * tau**-1 * x**(-1/4) * np.exp(-(1 + x**2) / tau) * iv(0.25, 2 * x / tau)


def main():
    parser = argparse.ArgumentParser(description="Plot data from HDF5 files.")
    parser.add_argument('filenames', nargs='+', help='List of HDF5 filenames to plot')
    args = parser.parse_args()

    for filename in args.filenames:
        with h5py.File(filename, 'r') as file:
            if 'chkpt' in filename:
                time = file['state']['time'][()]
                mass = file['state']['mass'][:]
                nu = file['config']['viscosity'][()]
                r0, r1, dlogr = file['config']['domain'][()]
                rf = np.logspace(np.log10(r0), np.log10(r1), len(mass) + 1)
                rc = 0.5 * (rf[:-1] + rf[1:])
                dA = np.pi * np.diff(rf**2)
                sigma = mass / dA
            elif 'prods' in filename:
                time = file['__time__'][()]
                nu = file['__config__']['viscosity'][()]
                rc = file['r'][()]
                sigma = file['sigma'][()]
            else:
                raise RuntimeError("must be a checkpoint or products file")

        plt.plot(rc, sigma, label=filename)

    # sigma_true = [ring_sigma(1.0, r, time, 1.0, nu) for r in rc]
    # plt.plot(rc, sigma_true, '--', c='k', lw=1.0)

    plt.xlabel(r'$r$')
    plt.ylabel(r'$\Sigma$')
    plt.grid(True)
    plt.legend()
    plt.show()


main()

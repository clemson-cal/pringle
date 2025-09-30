import argparse
import matplotlib.pyplot as plt
import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot data from a file.")
    parser.add_argument("filename", type=str, help="Path to the data file")
    args = parser.parse_args()

    filename = args.filename

    with h5py.File(filename, 'r') as file:
        time = file['timeseries']['time'][()]
        mass_accreted = file['timeseries']['mass_accreted'][()]
        mass_expelled = file['timeseries']['mass_expelled'][()]

    Mdot_accreted = np.diff(mass_accreted) / np.diff(time)
    Mdot_expelled = np.diff(mass_expelled) / np.diff(time)

    plt.plot(time[1:], Mdot_accreted, label=r"$\dot M_{accreted}$")
    plt.plot(time[1:], Mdot_expelled, label=r"$\dot M_{expelled}$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\dot M$")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

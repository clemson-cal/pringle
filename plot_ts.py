from pathlib import Path
from numpy import exp

ell = 1.0
nub = 1.0 / 6.0
G = 1.0
M = 1.0
Mdot_inf = 1.0


def nu(r):
    return 1 / 6


def a(t):
    return t ** (1 / 4)


def r_nu(t):
    return t ** (2 / 3)


def t_nu(r):
    return r ** (3 / 2)


def Mdot(t):
    return Mdot_inf * exp(-3 * ell / (5 * a(t) ** (5 / 6)))


def sigma_in(r, t):
    return Mdot(t) / (3 * pi * nu(r)) * (1.0 - ell * (a(t) / r) ** (1 / 2))


def sigma(r, t):
    return (r < r_nu(t)) * sigma_in(r, t) + (r >= r_nu(t)) * sigma_in(r, t_nu(r))


def r_star(t):
    return ell**2 * a(t)


def r_vac():
    return 0.0 if ell <= 0.0 else ell ** (16 / 5)


def main(filenames: list[Path]):
    from matplotlib import pyplot as plt
    from h5py import File

    fig, ax1 = plt.subplots()
    ax1.axhline(1.0, lw=1.0, ls="--", c="pink", label="Mass inflow rate")

    for filename in filenames:
        h5f = File(filename)
        time = h5f["timeseries"]["time"][...]
        mdot = h5f["timeseries"]["mdot"][...]
        ax1.plot(time, mdot, label=filename)

    # ax1.plot(time, Mdot(time), label="Fit", ls="--", lw=2.0, c="k")
    ax1.set_ylabel(r"$\dot M$")
    ax1.set_xlabel(r"Time")
    ax1.legend()
    plt.show()


if __name__ == "__main__":
    from typer import Typer

    app = Typer(pretty_exceptions_enable=False)
    app.command()(main)

    try:
        app()
    except RuntimeError as e:
        print("Error:", e)

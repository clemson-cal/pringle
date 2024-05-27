from os import getenv
from matplotlib.pyplot import figure, show, cm, rc
from numpy import pi, linspace, exp
from h5py import File

txt_width = 508 / 72 / 2 - 0.02
col_width = 244 / 72
ell = 1.0
nub = 1.0
G = 1.0
M = 1.0
Mdot_inf = 1.0
times = (1, 2, 4, 8, 16, 32)
hardcopy = getenv("HARDCOPY", "0").lower() in ("true", "1")


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


def main():
    if hardcopy:
        rc(group="text", usetex=True)

    fig = figure(figsize=[col_width, col_width * 0.75])
    ax1 = fig.add_subplot(111)

    for fname in ("prods.0050.h5",):
        with File(fname) as h5f:
            r = h5f["radius"][:]
            s = h5f["sigma"][:]
            f = h5f["mdot"][1:]
            t = h5f["__config__"]["tstart"] - h5f["__time__"][...]
            ax1.plot(r, f, label=fname)

        rmin = max(0.1, r_star(t), r_vac())
        print("simulation time:", t)
        print("mdot:", Mdot(t))
        print("r_star:", r_star(t))
        print("r_vac:", r_vac())
        r = linspace(rmin, r[-1], 10000)
        ax1.plot(r, sigma(r, t), lw=2, c="k", ls="--", alpha=0.7, label=rf"$t={t}$")
        ax1.plot(r_nu(t), sigma(r_nu(t), t), marker="o", mfc="none", ms=10)

    ax1.legend(ncol=2)
    ax1.set_xlabel(r"Radius $[r_{\rm dec}]$")
    ax1.set_ylabel(r"Surface density $\Sigma(r)$")

    if hardcopy:
        fig.tight_layout(pad=0.1)
        fig.savefig(__file__.replace(".py", ".pdf"))
    else:
        show()


if __name__ == "__main__":
    main()

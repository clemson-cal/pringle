from pathlib import Path


def get_field(h5f, field):
    if field == "jdot":
        return h5f["jdot_viscosity"][...] + h5f["jdot_advection"][...]
    elif field == "jdot_viscosity_neg":
        return h5f["jdot_viscosity"][...] * -1.0
    elif field == "one_minus_mdot":
        return 1.0 - get_field(h5f, "mdot") / get_field(h5f, "mdot")[-1]
    elif field == "one_minus_jdot":
        return 1.0 - get_field(h5f, "jdot") / get_field(h5f, "jdot")[-1]
    elif field == "jdot_over_mdot":
        return get_field(h5f, "jdot") / get_field(h5f, "mdot")
    else:
        return h5f[field][...]


def main(
    filenames: list[Path],
    x: str = "r",
    field: str = "sigma",
    log: bool = False,
):
    from numpy import argmin
    from matplotlib import pyplot as plt
    from h5py import File

    fig, ax1 = plt.subplots()
    for filename in filenames:
        for f in field.split(","):
            h5f = File(filename)
            r = h5f["radius"][...]
            t = h5f["__config__"]["tstart"][...] - h5f["__time__"][...]
            l = r**0.5
            y = get_field(h5f, f)[: r.shape[0]]
            if x == "r":
                ax1.plot(r, y, label=str(filename) + "/" + f)
            if x == "l":
                ax1.plot(l, y, label=str(filename) + "/" + f)
            # ax1.axhline(t**0.125, ls="--")
            # i = argmin(abs(t**0.333 - l))
            # ax1.scatter(l[i], y[i])

    if x == "r":
        ax1.set_xlabel(r"Radius $r$")
    if x == "l":
        ax1.set_xlabel(r"Specific Angular Momentum $\ell$")
    if log:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
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

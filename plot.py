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
    else:
        return h5f[field][...]


def main(filenames: list[Path], field: str = "sigma", log: bool = False):
    from matplotlib import pyplot as plt
    from h5py import File

    fig, ax1 = plt.subplots()
    for filename in filenames:
        for f in field.split(","):
            h5f = File(filename)
            r = h5f["radius"][...]
            y = get_field(h5f, f)[: r.shape[0]]
            ax1.plot(r, y, label=str(filename) + "/" + f)

    if log:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.legend()
    ax1.set_xlabel("Radius")
    plt.show()


if __name__ == "__main__":
    from typer import Typer

    app = Typer(pretty_exceptions_enable=False)
    app.command()(main)

    try:
        app()
    except RuntimeError as e:
        print("Error:", e)

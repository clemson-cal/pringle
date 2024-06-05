from pathlib import Path
from numpy import exp


def mdot_model(t, n: float = 0.0, tau: float = 1.0, kappa: float = 1.0):
    from numpy import nan_to_num

    tau1 = tau
    tau2 = tau / kappa
    p = (2 + n) / (2 - n)
    q = (2 + n) / 4
    pre = (1 - (+t / tau1) ** (-p / 8)) ** (1 / p)
    pos = (1 - (-t / tau2) ** (-p / 8)) ** (1 / q)
    return (t > tau1) * nan_to_num(pre) + (-t > tau2) * nan_to_num(pos)


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
        n = h5f["config"]["n"][...]
    ax1.plot(time, mdot_model(time, n=n), label="Fit", ls="--", lw=2.0, c="k")
    ax1.invert_xaxis()
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

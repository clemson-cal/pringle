from pathlib import Path


def main(filenames: list[Path]):
    from matplotlib import pyplot as plt
    from matplotlib import rc
    from h5py import File

    # rc(group="text", usetex=True)
    fig, ax1 = plt.subplots()
    ax1.axhline(1.0, lw=1.0, ls="--", c="pink", label="Mass inflow rate")
    for filename in filenames:
        h5f = File(filename)
        ir = h5f["config"]["inspiral_rate"][...]
        if ir > 0.0:
            ax1.axvline(ir**-1, c="grey", ls="--", lw=1.0, label="Merger time")
        time = h5f["timeseries"]["time"][...]
        mdot = h5f["timeseries"]["mdot"][...]
        ax1.plot(time, mdot - 0, label=filename)
    ax1.set_ylim(0.0, 1.5)
    ax1.set_xlabel("Time [Initial Binary Orbits]")
    ax1.set_ylabel(r"$\dot M$")
    ax1.legend()
    # plt.savefig("samuel.pdf")
    plt.show()


if __name__ == "__main__":
    from typer import Typer

    app = Typer(pretty_exceptions_enable=False)
    app.command()(main)

    try:
        app()
    except RuntimeError as e:
        print("Error:", e)

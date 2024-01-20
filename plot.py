from pathlib import Path


def main(filenames: list[Path]):
    from matplotlib import pyplot as plt
    from h5py import File

    fig, ax1 = plt.subplots()
    for filename in filenames:
        h5f = File(filename)
        r = h5f["radius"][...]
        s = h5f["sigma"][...]
        ax1.plot(r, s, label=filename)
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

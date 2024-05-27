#!/usr/bin/env python3

from enum import StrEnum, auto
from numpy import *
from matplotlib import pyplot as plt
import math

GM = 1.0


class PlotField(StrEnum):
    vr = auto()
    sigma = auto()
    mdot = auto()
    jdot = auto()


def plot_field(ax, M, R, viscosity, field: PlotField, **kwargs):
    R0 = R[:-1]
    R1 = R[+1:]
    Rc = 0.5 * (R0 + R1)
    kw = dict(ls="-", lw=2, marker=None, mfc=None)
    kw.update(kwargs)

    c = {
        "blue": "#1f77b4",  # matplotlib.colors.TABLEAU_COLORS
        "orange": "#ff7f0e",
        "green": "#2ca02c",
        "red": "#d62728",
        "purple": "#9467bd",
        "brown": "#8c564b",
        "pink": "#e377c2",
        "gray": "#7f7f7f",
        "olive": "#bcbd22",
        "cyan": "#17becf",
    }
    match field:
        case PlotField.vr:
            vr = radial_velocity(M, R)
            ax.plot(R[1:-1], vr, label=r"$v_R$", c=c["blue"], **kw)
        case PlotField.sigma:
            s = sigma(M, R)
            # ax.plot(Rc / 10, s * pi * 100, label=r"$\Sigma$", c=c["brown"], **kw)
            # plt.plot(Rc / 10, s * pi * 100, color="red")
            plt.plot(Rc, s, color="red", alpha=0.4)
            plt.title(f"Sigma Profile")
            plt.xlabel("X")
            plt.ylabel("Sigma")
            # plot_solutions_exp_x.append(Rc / 10)
            # plot_solutions_exp_y.append(s)

        case PlotField.mdot:
            md = godunov_fluxes(M, R, viscosity)
            ax.plot(R[1:-1], md[1:-1], label=r"$\dot M$", c=c["gray"], **kw)
        case PlotField.jdot:
            md, jd1, jd2 = godunov_fluxes(M, R, viscosity, ret_torque=True)
            jd0 = jd1 + jd2
            ax.plot(R[1:-1], jd0[1:-1], label=r"$\dot J$", c=c["purple"], **kw)
            ax.plot(R[1:-1], jd1[1:-1], label=r"$\dot J_{\rm adv}$", c=c["olive"], **kw)
            ax.plot(R[1:-1], jd2[1:-1], label=r"$\dot J_{\rm visc}$", c=c["cyan"], **kw)

    # ax.set_xlabel(r"$R$")
    # ax.legend()
    # ax.set_yscale("log")


def show_plot(M, R, t, model, fields: list[PlotField], plot_analytic):
    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    m = 1
    R_0 = 10
    # time_end
    x = R / 10

    for field in fields:
        plot_field(ax1, M, R, model.viscosity, field)

    if plot_analytic:
        dR = diff(R)
        R0 = R[:-1]
        R1 = R[+1:]
        Rc = 0.5 * (R0 + R1)
        s = model.sigma(Rc, t)
        M = s * pi * (R1**2 - R0**2)

        for field in fields:
            plot_field(ax1, M, R, model.viscosity, field, ls="--", lw=1.0)

    from scipy.special import iv

    sol_plot = []
    """
    a = 1.0 / (pi * 10**2)
    b = 1.0 / (time_test * x **.25)
    c = exp(-(1 + x**2) / time_test)
    d = iv(.25, 2 * x / time_test)
    d[isinf(d)] = 0.0
    s = m * a * b * c * d
    s[s < 1e-12] = 1e-12
    """
    # plt.plot(x, s * pi * 100, '--', color="black", alpha=.7)   #Pringle solution
    plt.legend(["Sigma"])

    # plot_solutions_true_x.append(x)
    # plot_solutions_true_y.append(s * pi * 100)


def keplerian_omega_log_derivative(R):
    """
    Return the shear profile d(Omega) / d(log R)
    """
    return -1.5 * sqrt(GM / R**3)


def keplerian_omega(R):
    return sqrt(GM / R**3)


def specific_angular_momentum(R):
    return sqrt(GM * R)


def specific_angular_momentum_derivative(R):
    return 0.5 * sqrt(GM / R)


def sigma(M, R):
    """
    Return the surface density
    """
    return M / (pi * diff(R**2))


def radial_velocity(M, R, viscosity, model, t):
    """
    Return the radial gas velocity at the internal zone interfaces

    v = (d/dR(R g) + tau) / (sigma R l')

    where tau is the external torque per unit length.
    """
    Rc = 0.5 * (R[1:] + R[:-1])
    s = sigma(M, R)
    n = viscosity(Rc)
    A = keplerian_omega_log_derivative(Rc)
    g = Rc * s * n * A
    m = specific_angular_momentum_derivative(R[1:-1])

    print(diff(g * Rc))

    try:
        tau = model.external_torque_per_unit_length(M, R, t)
    except AttributeError:
        tau = 0.0

    return (diff(Rc * g) / diff(Rc) + tau) / (0.5 * (s[1:] + s[:-1]) * m * R[1:-1])


def godunov_fluxes(M, R, viscosity, model, t, ret_torque=False):
    R0 = R[:-1]
    R1 = R[+1:]
    Fhat = zeros_like(R)
    s = sigma(M, R)
    v_hat = radial_velocity(M, R, viscosity, model, t)
    s_hat = (v_hat > 0.0) * s[:-1] + (v_hat < 0.0) * s[1:]
    Fhat[1:-1] = 2 * pi * R[1:-1] * s_hat * v_hat

    if ret_torque:
        A = keplerian_omega_log_derivative(R)
        nu = viscosity(R)
        Ghat = zeros_like(R)
        Ghat[1:-1] = -2 * pi * s_hat * (nu * A * R**2)[1:-1]
        return Fhat, Fhat * specific_angular_momentum(R), Ghat
    else:
        return Fhat


def time_derivative(M, R, t, model):
    Fhat = godunov_fluxes(M, R, model.viscosity, model, t)
    model.flux_boundary_condition(Fhat)
    L = -diff(Fhat)

    try:
        L += model.mass_source_term(M, R, t)
    except AttributeError:
        pass

    return L


class SteadyDisk:
    def __init__(self, Mdot=-1.0, Jdot=0.0):
        self.Mdot = Mdot
        self.Jdot = Jdot
        self.nu = 0.001

    def sigma(self, R, t):
        Mdot = self.Mdot
        Jdot = self.Jdot
        nu = self.viscosity(R)
        j = specific_angular_momentum(R)
        s = (Jdot - Mdot * j) / (3 * pi * nu * j)
        return s

    def viscosity(self, R):
        return self.nu * ones_like(R)

    def flux_boundary_condition(self, Fhat):
        Fhat[-1] = self.Mdot
        Fhat[+0] = self.Mdot


class BinaryInspiral:
    def __init__(self, Mdot=-1.0, Jdot=0.0):
        self.Mdot = Mdot
        self.Jdot = Jdot
        self.nu = 0.001

    def sigma(self, R, t):
        Mdot = self.Mdot
        Jdot = self.Jdot
        nu = self.viscosity(R)
        j = specific_angular_momentum(R)
        s = (Jdot - Mdot * j) / (3 * pi * nu * j)
        return s

    def viscosity(self, R):
        return self.nu * ones_like(R)

    def flux_boundary_condition(self, Fhat):
        Fhat[-1] = self.Mdot
        Fhat[+0] = 0.0

    def mass_source_term(self, M, R, t):
        Rc = 0.5 * (R[1:] + R[:-1])
        a = 1.0
        omega = keplerian_omega(maximum(Rc, 0.05))
        return -0.1 * M * omega * exp(-((Rc / a) ** 2))

    def external_torque_per_unit_length(self, M, R, t):
        """
        Return the torque per unit length, at faces
        """
        dM_dR_p = M[+1:] / diff(R[+1:])
        dM_dR_m = M[:-1] / diff(R[:-1])
        dM_dR = 0.5 * (dM_dR_m + dM_dR_p)
        dL_dR = dM_dR * specific_angular_momentum(R[1:-1])
        a = 1.0
        f = 0.05 * exp(-((R[1:-1] / a) ** 2))
        omega = keplerian_omega(a)
        return dL_dR * omega * f


class DiskModel(StrEnum):
    steady = auto()
    ring = auto()
    inspiral = auto()

    def create_with(self, **parameters):
        match self:
            case DiskModel.steady:
                return SteadyDisk(**parameters)
            case DiskModel.ring:
                return SpreadingRing(**parameters)
            case DiskModel.inspiral:
                return BinaryInspiral(**parameters)


def key_val(s):
    result = s.split("=")
    if len(result) != 2:
        raise TypeError("must have the form key=val")
    return result[0], eval(result[1])


def main(
    model: DiskModel,
    params: list[str] = list(),
    domain: tuple[float, float, int] = (0.0, 50.0, 10000),
    trange: tuple[float, float] = (0, 50),  # If ring, start from .3 not 0
    plot: list[PlotField] = list(),
    plot_analytic: bool = False,
):
    try:
        model = model.create_with(**dict(map(key_val, params)))
    except Exception as e:
        raise RuntimeError(e)

    R = linspace(domain[0], domain[1], domain[2] + 1)
    dR = diff(R)
    R0 = R[:-1]
    R1 = R[+1:]
    Rc = 0.5 * (R0 + R1)
    nu = model.viscosity(Rc)
    dt = 0.1 * (dR**2 / nu).min()

    t = trange[0]
    n = 0
    s = model.sigma(Rc, t)
    M = s * pi * (R1**2 - R0**2)

    if (s < 0.0).any():
        R0 = R[:-1][s < 0].min()
        R1 = R[+1:][s < 0].max()
        raise ValueError(f"no disk solution for R between {R0:0.3f} and {R1:0.3f}")

    while t < trange[1]:
        K = time_derivative(M, R, t, model)
        M += K * dt
        t += dt
        n += 1
        print(f"[{n:04d}] t={t:04f}       ")

    show_plot(M, R, t, model, plot, plot_analytic)
    plt.show()


if __name__ == "__main__":
    from typer import Typer

    app = Typer(pretty_exceptions_enable=False)
    app.command()(main)

    try:
        app()
    except RuntimeError as e:
        print("Error:", e)

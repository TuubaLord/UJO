"""
fluid_film_bearing.py
=====================

This module implements a simple numerical model for the pressure and shear
stress distribution in a hydrodynamic fluid‐film (journal) bearing.  The
governing equation is the Reynolds equation, which describes the pressure
distribution in a thin fluid film.  For an incompressible lubricant the
Reynolds equation simplifies because the density of the fluid is constant;
for a compressible lubricant the density can vary with pressure and an
appropriate equation of state must be used.  The model in this script
implements both cases in the framework of a one–dimensional circumferential
domain, appropriate for a long journal bearing where axial variations are
negligible.

The pressure distribution is obtained by discretising the circumferential
coordinate into `N` points and performing a Gauss–Seidel iteration over
the discrete form of the generalised Reynolds equation.  Cavitation is
handled by enforcing a non–negative pressure at each iteration (the
Half–Sommerfeld boundary condition).  Once the pressure has converged,
the shear stress on the bearing surface is computed from the local
velocity gradient assuming Couette flow.

The module can be executed as a script.  When run directly it will
compute the pressure and shear stress for both compressible and
incompressible fluids, generate plots of the pressure and shear stress
versus the circumferential angle, and save the figure to a PNG file.  The
figure illustrates the influence of compressibility on the hydrodynamic
pressure build–up in a rotating bearing.

References
----------
* Osborne Reynolds, 1886.  **On the Theory of Lubrication and Its
  Application to Mr. Beauchamp Tower’s Experiments**.  Philosophical
  Transactions of the Royal Society of London.  The Reynolds equation was
  originally derived assuming an iso–viscous, incompressible fluid.

* The generalised Reynolds equation includes fluid density as a variable
  to account for compressibility.  In a review of the derivation
  [Miyan 2016], it is noted that the original Reynolds equation was
  restricted to incompressible fluids and was later generalised to
  include compressibility effects by incorporating the density into the
  equation of motion and continuity equations【871049173512994†L610-L617】.
  The generalised equation can be written in one dimension as
  \( \frac{\mathrm{d}}{\mathrm{d}\theta}(\rho h^3\frac{\mathrm{d}p}{\mathrm{d}\theta}) = 6 \mu R\omega\,\frac{\mathrm{d}(\rho h)}{\mathrm{d}\theta} \).

* For incompressible fluids, the density is constant and drops out of
  the continuity equation; for compressible fluids the density appears
  explicitly, making the continuity equation dynamic【871049173512994†L371-L383】.
  Constant compressibility is modelled here through an exponential
  equation of state \(\rho = \rho_0 \exp(C(p-p_0))\)【871049173512994†L241-L268】.

"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class BearingParameters:
    """Container class for bearing parameters.

    Attributes
    ----------
    R : float
        Radius of the journal [m].
    c : float
        Radial clearance between journal and bearing housing [m].
    e : float
        Eccentricity of the journal centre relative to the bearing centre [m].
    mu : float
        Dynamic viscosity of the lubricant [Pa·s].
    rho0 : float
        Reference density of the lubricant at ambient pressure [kg/m³].
    compressibility : float
        Compressibility coefficient C [1/Pa]; defines \(\rho = \rho_0 e^{C(p-p_0)}\).
    omega : float
        Angular speed of the journal [rad/s].
    p0 : float
        Ambient (reference) pressure [Pa]; used in the compressible case.
    N : int
        Number of discrete points along the circumferential direction.
    max_iter : int
        Maximum number of Gauss–Seidel iterations.
    tol : float
        Convergence tolerance for the relative change of the pressure field.
    """

    R: float = 0.05  # 50 mm radius
    c: float = 100e-6  # 100 µm radial clearance
    # Eccentricity of the journal centre [m].  A smaller eccentricity
    # reduces the minimum film thickness and thus lowers the load support.
    e: float = 30e-6   # 30 µm eccentricity
    mu: float = 0.03   # dynamic viscosity [Pa·s] (typical for lubricating oil)
    rho0: float = 850.0  # density of lubricant [kg/m^3]
    # Compressibility coefficient C [1/Pa].  A smaller value indicates a
    # stiffer (less compressible) fluid.  Typical lubricants have C on the
    # order of 10⁻⁹ Pa⁻¹.  Large values can lead to slow convergence in
    # the numerical solver.
    compressibility: float = 1e-9
    # Angular speed of the journal [rad/s].  Reducing the speed decreases
    # the entraining velocity and improves numerical stability for the
    # compressible case.
    omega: float = 100.0  # angular speed [rad/s] ≈ 955 rpm
    p0: float = 0.0     # ambient pressure (gauge) [Pa]
    N: int = 200       # number of grid points along θ
    # Increase the maximum iteration count for the Gauss–Seidel solver.  The
    # compressible Reynolds equation can require many iterations to converge
    # because the density and pressure fields are coupled.  A larger
    # iteration limit helps to obtain convergence in most cases.
    max_iter: int = 20000
    # Absolute convergence tolerance.  The iterative solver stops when
    # the maximum change in pressure between successive iterations is
    # below this threshold.  Larger values speed up convergence at the
    # cost of accuracy.  A value of 1e-3 is sufficient for illustrative
    # purposes.
    tol: float = 1e-3

    @property
    def epsilon(self) -> float:
        """Return the eccentricity ratio e/c."""
        return self.e / self.c if self.c != 0 else 0.0

    @property
    def U(self) -> float:
        """Return the sliding velocity of the journal relative to the housing [m/s]."""
        return self.R * self.omega


def _film_thickness(theta: np.ndarray, params: BearingParameters) -> np.ndarray:
    """Compute the film thickness h as a function of circumferential coordinate θ.

    The journal bearing has a clearance `c` and eccentricity `e`, leading to a
    film thickness that varies with the angular coordinate.  The film thickness
    is given by

    .. math::

        h(\theta) = c \bigl(1 + \epsilon \cos\theta\bigr),

    where \(\epsilon = e / c\) is the eccentricity ratio.  If the eccentricity
    ratio is zero the film thickness is constant.

    Parameters
    ----------
    theta : ndarray
        Array of circumferential angles in radians.
    params : BearingParameters
        Bearing parameter object.

    Returns
    -------
    ndarray
        Film thickness for each angle.
    """
    return params.c * (1.0 + params.epsilon * np.cos(theta))


def _equation_of_state(p: np.ndarray, params: BearingParameters) -> np.ndarray:
    """Compute density as a function of pressure using an exponential equation of state.

    For liquids with constant compressibility under isothermal conditions
    the density can be expressed as

    .. math::

        \rho(p) = \rho_0 \exp\bigl[C\,(p - p_0)\bigr],

    where `rho0` is the density at reference pressure `p0` and `C` is the
    compressibility coefficient【871049173512994†L241-L268】.  For an
    incompressible fluid the density is constant and equal to `rho0`.

    Parameters
    ----------
    p : ndarray
        Pressure array [Pa].
    params : BearingParameters
        Bearing parameter object.

    Returns
    -------
    ndarray
        Density values corresponding to each pressure value.
    """
    return params.rho0 * np.exp(params.compressibility * (p - params.p0))


def solve_reynolds(params: BearingParameters, compressible: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the (compressible or incompressible) Reynolds equation for a journal bearing.

    The numerical solution uses a finite difference discretisation of the one–
    dimensional generalised Reynolds equation.  For an incompressible fluid
    (constant density) the generalised Reynolds equation reduces to

    .. math::

        \frac{\mathrm{d}}{\mathrm{d}\theta}\bigl(h^3\frac{\mathrm{d}p}{\mathrm{d}\theta}\bigr)
        = 6\mu R\omega\,\frac{\mathrm{d}h}{\mathrm{d}\theta},

    whereas for a compressible fluid the density \(\rho\) multiplies the
    film–thickness terms and appears on the right–hand side:

    .. math::

        \frac{\mathrm{d}}{\mathrm{d}\theta}\bigl(\rho h^3\frac{\mathrm{d}p}{\mathrm{d}\theta}\bigr)
        = 6\mu R\omega\,\frac{\mathrm{d}(\rho h)}{\mathrm{d}\theta}
        .

    The discretisation leads to a tridiagonal system which is solved
    iteratively using a Gauss–Seidel scheme with cavitation (non–negative
    pressure) enforcement.  Periodic boundary conditions are applied in
    the circumferential direction.  Convergence is tested against a relative
    tolerance `params.tol`.

    Parameters
    ----------
    params : BearingParameters
        Bearing parameter object containing all physical and numerical
        parameters.
    compressible : bool, optional
        If `True`, solve the compressible Reynolds equation; otherwise
        solve the incompressible version.  Default is `False`.

    Returns
    -------
    p : ndarray
        Pressure distribution [Pa] over the circumferential grid.
    theta : ndarray
        Corresponding circumferential angles [rad].
    h : ndarray
        Film thickness distribution [m].
    """
    N = params.N
    theta = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
    dtheta = 2.0 * math.pi / N

    # Film thickness and its derivative
    h = _film_thickness(theta, params)
    # derivative dh/dtheta: central difference with periodic wrap
    dh_dtheta = params.epsilon * (-params.c) * np.sin(theta)

    # Initialise pressure (start with zero gauge pressure)
    p = np.zeros_like(theta)

    # Precompute constant factor in source term
    factor = 6.0 * params.mu * params.R * params.omega

    # Relaxation parameter for Gauss–Seidel iterations.  A value less than 1
    # under–relaxes the update and can improve convergence for stiff or
    # compressible problems.
    relaxation: float = 0.6

    for iteration in range(params.max_iter):
        p_old = p.copy()
        # Update density and g1 = rho * h^3, g = rho * h
        if compressible:
            rho = _equation_of_state(p, params)
        else:
            rho = np.full_like(p, params.rho0)
        g1 = rho * h**3
        g  = rho * h

        # Gauss–Seidel sweep over all points
        for i in range(N):
            ip = (i + 1) % N  # next index (periodic)
            im = (i - 1) % N  # previous index (periodic)
            g1_ip_half = 0.5 * (g1[i] + g1[ip])
            g1_im_half = 0.5 * (g1[im] + g1[i])

            # Source term S_i = factor * (g_ip - g_im) / (2*dtheta)
            S_i = factor * (g[ip] - g[im]) / (2.0 * dtheta)

            # Discrete equation:
            # g1_ip_half*(p[ip]-p[i]) - g1_im_half*(p[i]-p[im]) = S_i*dtheta**2
            # Rearranged for p[i]:
            denominator = g1_ip_half + g1_im_half
            numerator = g1_ip_half * p[ip] + g1_im_half * p[im] - S_i * dtheta**2
            p_raw = numerator / denominator if denominator != 0 else 0.0

            # Cavitation: enforce non–negative pressure on the raw value
            if p_raw < 0.0:
                p_raw = 0.0

            # Under–relaxation update
            p_new = (1.0 - relaxation) * p[i] + relaxation * p_raw
            # Additional cavitation enforcement after relaxation
            if p_new < 0.0:
                p_new = 0.0
            p[i] = p_new

        # Convergence check.  We use an absolute tolerance on the maximum
        # change in the pressure field.  A relative tolerance based on the
        # previous pressure magnitude can be overly stringent when the
        # pressures start from zero.  Convergence is declared when the
        # maximum pointwise change drops below `params.tol`.
        max_change = np.max(np.abs(p - p_old))
        if max_change < params.tol:
            break
    else:
        print(f"Warning: solution did not converge after {params.max_iter} iterations.")

    return p, theta, h


def compute_shear_stress(h: np.ndarray, params: BearingParameters) -> np.ndarray:
    """Compute the shear stress on the bearing surface.

    Assuming Couette flow between the journal and housing, the shear stress
    on the bearing surface is given by

    .. math::

        \tau(\theta) = \mu \frac{U}{h(\theta)},

    where \(U = R\,\omega\) is the sliding speed of the journal【871049173512994†L610-L617】.
    This expression neglects any velocity slip at the surface and assumes
    a linear velocity profile across the film thickness.

    Parameters
    ----------
    h : ndarray
        Film thickness distribution [m].
    params : BearingParameters
        Bearing parameters, providing viscosity and sliding speed.

    Returns
    -------
    ndarray
        Shear stress distribution [Pa] at each circumferential grid point.
    """
    return params.mu * params.U / h


def main() -> None:
    """Run a demonstration of the fluid film bearing model.

    This function sets up a default bearing, solves the Reynolds equation for
    both incompressible and compressible fluids, computes the shear stress,
    and produces a plot of the resulting pressure and shear stress
    distributions versus circumferential angle.  The resulting figure is
    saved in the working directory as ``fluid_film_bearing_results.png``.
    """
    # Set up default parameters (the dataclass provides reasonable defaults)
    params = BearingParameters()

    # Solve for incompressible and compressible cases
    p_incompressible, theta, h = solve_reynolds(params, compressible=False)
    p_compressible, _, _ = solve_reynolds(params, compressible=True)

    # Compute shear stress for the incompressible case
    shear_incompressible = compute_shear_stress(h, params)

    # Convert theta to degrees for plotting
    theta_deg = np.degrees(theta)

    # Plot results
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Journal bearing pressure and shear stress distribution')

    # Pressure distribution
    ax[0].plot(theta_deg, p_incompressible / 1e3, label='Incompressible', color='tab:blue')
    ax[0].plot(theta_deg, p_compressible / 1e3, label='Compressible', color='tab:red', linestyle='--')
    ax[0].set_ylabel('Pressure [kPa]')
    ax[0].set_title('Pressure distribution along circumferential angle')
    ax[0].legend(loc='best')
    ax[0].grid(True)

    # Shear stress distribution (incompressible case)
    ax[1].plot(theta_deg, shear_incompressible / 1e3, label='Shear stress', color='tab:green')
    ax[1].set_xlabel('Angle θ [°]')
    ax[1].set_ylabel('Shear stress [kPa]')
    ax[1].set_title('Shear stress distribution (incompressible fluid)')
    ax[1].grid(True)

    # Improve layout and save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('fluid_film_bearing_results.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
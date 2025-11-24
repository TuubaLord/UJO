# reynolds_1d_analytic.py
import numpy as np

def slider_geometry_h(x, k, h_T, Lx):
    """
    Linear slider film thickness (Almqvist Eqs. 5.96–5.97):

        h(x) = h_T * (1 + k - k x / Lx)

    Parameters
    ----------
    x : array_like
        Positions along the slider (0 <= x <= Lx).
    k : float
        Inclination parameter.
    h_T : float
        Trailing edge thickness [m].
    Lx : float
        Slider length [m].

    Returns
    -------
    h : ndarray
        Film thickness h(x).
    """
    x = np.asarray(x)
    return h_T * (1.0 + k - k * x / Lx)


def analytic_pressure_linear_slider(x, k, mu_a, u_l, h_T, Lx):
    """
    Analytical 1D pressure for the linear slider (Almqvist Eq. 5.106).

    Parameters
    ----------
    x : array_like
        Positions along the slider (0 <= x <= Lx).
    k : float
        Inclination parameter.
    mu_a : float
        Dynamic viscosity μ_a [Pa·s].
    u_l : float
        Sliding speed of the lower surface [m/s].
    h_T : float
        Trailing edge thickness [m].
    Lx : float
        Slider length [m].

    Returns
    -------
    p : ndarray
        Analytical pressure p(x) [Pa].
    """
    x = np.asarray(x)
    H = 1.0 + k - k * x / Lx
    bearing_number = 6.0 * mu_a * u_l * Lx / (h_T ** 2)
    shape = (1.0 / k) * (
        1.0 / H
        - (1.0 + k) / (2.0 + k) / (H ** 2)
        - 1.0 / (2.0 + k)
    )
    return bearing_number * shape

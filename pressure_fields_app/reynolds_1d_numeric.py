# reynolds_1d_numeric.py
import numpy as np

def solve_reynolds_1d_incompressible(x, h, u_l, mu_a):
    """
    Finite-difference solution of 1D stationary Reynolds equation
    (Almqvist Eq. 5.99):

        d/dx[ (h^3 / (12 μ_a)) dp/dx ] = (u_l / 2) dh/dx,
        p(0) = p(Lx) = 0

    Parameters
    ----------
    x : ndarray (N,)
        Grid points along the slider [m].
    h : ndarray (N,)
        Film thickness h(x) [m].
    u_l : float
        Sliding speed of lower surface [m/s].
    mu_a : float
        Dynamic viscosity μ_a [Pa·s].

    Returns
    -------
    p : ndarray (N,)
        Numerical pressure p(x) [Pa].
    """
    x = np.asarray(x)
    h = np.asarray(h)
    N = x.size
    if N < 3:
        raise ValueError("Need at least 3 grid points for 1D Reynolds solver.")
    dx = x[1] - x[0]

    # Coefficient c(x) = h^3 / (12 μ_a)
    c = h**3 / (12.0 * mu_a)

    # Interior unknowns 1..N-2
    M = N - 2
    A = np.zeros((M, M), dtype=float)
    b = np.zeros(M, dtype=float)

    for j in range(M):
        i = j + 1  # global index in [1, N-2]

        c_imh = 0.5 * (c[i - 1] + c[i])
        c_iph = 0.5 * (c[i] + c[i + 1])

        A[j, j] = (c_imh + c_iph) / dx**2
        if j > 0:
            A[j, j - 1] = -c_imh / dx**2
        if j < M - 1:
            A[j, j + 1] = -c_iph / dx**2

        dhdx_i = (h[i + 1] - h[i - 1]) / (2.0 * dx)
        b[j] = - (u_l / 2.0) * dhdx_i  # sign from Eq. 5.99

    p_int = np.linalg.solve(A, b)

    p = np.zeros_like(h)
    p[1:-1] = p_int
    p[0] = 0.0
    p[-1] = 0.0
    return p

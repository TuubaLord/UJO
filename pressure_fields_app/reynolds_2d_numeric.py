# reynolds_2d_numeric.py
import numpy as np

def solve_reynolds_2d_incompressible(h, dx, dy, u_l, mu_a,
                                     max_iter=5000, tol=1e-5, omega=1.6,
                                     bc_y="neumann"):
    """
    Solve 2D incompressible Reynolds equation on a rectangular grid:

        ∂x ( c ∂x p ) + ∂y ( c ∂y p ) = (u_l / 2) ∂x h,
        c = h^3 / (12 μ_a)

    Boundary conditions
    -------------------
    x = 0, x = Lx : p = 0  (Dirichlet)
    y = 0, y = Ly :
        bc_y = "dirichlet" → p = 0
        bc_y = "neumann"   → ∂p/∂y = 0 (implemented by copying interior rows)

    Parameters
    ----------
    h : ndarray (Ny, Nx)
        Film thickness h(x, y) [m].
    dx, dy : float
        Grid spacing in x and y [m].
    u_l : float
        Sliding speed of lower surface in +x direction [m/s].
    mu_a : float
        Dynamic viscosity μ_a [Pa·s].
    max_iter : int
        Maximum number of SOR iterations.
    tol : float
        Convergence tolerance on max update |p_new - p_old|.
    omega : float
        SOR relaxation parameter (1.0 = Gauss–Seidel).
    bc_y : {"neumann", "dirichlet"}
        Type of boundary condition in y.

    Returns
    -------
    p : ndarray (Ny, Nx)
        Pressure field p(x, y) [Pa].
    """
    h = np.asarray(h)
    Ny, Nx = h.shape
    p = np.zeros_like(h)
    c = h**3 / (12.0 * mu_a)

    for _ in range(max_iter):
        max_res = 0.0

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                c_imh = 0.5 * (c[j, i - 1] + c[j, i])
                c_iph = 0.5 * (c[j, i] + c[j, i + 1])
                c_jmh = 0.5 * (c[j - 1, i] + c[j, i])
                c_jph = 0.5 * (c[j, i] + c[j + 1, i])

                dhdx = (h[j, i + 1] - h[j, i - 1]) / (2.0 * dx)
                rhs = (u_l / 2.0) * dhdx

                p_im = p[j, i - 1]
                p_ip = p[j, i + 1]
                p_jm = p[j - 1, i]
                p_jp = p[j + 1, i]

                Ax = (c_iph + c_imh) / dx**2
                Ay = (c_jph + c_jmh) / dy**2
                A_center = Ax + Ay

                p_new = (
                    c_iph * p_ip / dx**2
                    + c_imh * p_im / dx**2
                    + c_jph * p_jp / dy**2
                    + c_jmh * p_jm / dy**2
                    - rhs
                ) / A_center

                p_old = p[j, i]
                p[j, i] = (1.0 - omega) * p_old + omega * p_new
                max_res = max(max_res, abs(p[j, i] - p_old))

        # x-boundaries: Dirichlet p=0
        p[:, 0] = 0.0
        p[:, -1] = 0.0

        # y-boundaries
        if bc_y == "dirichlet":
            p[0, :] = 0.0
            p[-1, :] = 0.0
        else:  # "neumann"
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]

        if max_res < tol:
            break

    return p

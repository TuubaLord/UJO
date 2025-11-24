"""
Simple 2D Reynolds-equation example based on Almqvist, Chapter 5.

Geometry:
    Rectangular domain, 0 <= x <= Lx, 0 <= y <= Ly
    Film thickness from the linear slider (Eqs. (5.96)–(5.97)):
        h(x) = h_T (1 + k - k x / Lx),  independent of y.

Physics:
    Incompressible, iso-viscous Reynolds equation (Section 5.2):
        ∂/∂x( c ∂p/∂x ) + ∂/∂y( c ∂p/∂y ) = - (u_l / 2) ∂h/∂x
        where c = h^3 / (12 μ_a)

Boundary conditions:
    p = 0 on all boundaries (ambient pressure).

We solve the 2D problem with a simple Gauss–Seidel/SOR iteration,
and compare the mid-line p(x, y = Ly/2) against the 1D analytical
solution from Eq. (5.106) (Section 5.3.1).
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Geometry & analytical 1D solution (same as before)
# ---------------------------------------------------------------------
def slider_geometry_h(x, k, h_T, Lx):
    """Film thickness h(x) for infinitely wide linear slider.

    Almqvist Eqs. (5.96)-(5.97):
        h(x) = h_T (1 + k - k x / Lx)
    """
    return h_T * (1 + k - k * x / Lx)


def analytic_pressure_linear_slider(x, k, mu_a, u_l, h_T, Lx):
    """Analytical 1D pressure solution (Eq. 5.106, Section 5.3.1)."""
    H = 1 + k - k * x / Lx
    bearing_number = 6 * mu_a * u_l * Lx / (h_T**2)  # Eq. (5.107)
    shape = (1 / k) * (1 / H - (1 + k) / (2 + k) / H**2 - 1 / (2 + k))
    return bearing_number * shape


# ---------------------------------------------------------------------
# 2D Reynolds solver (Gauss–Seidel / SOR)
# ---------------------------------------------------------------------
def solve_reynolds_2d_incompressible(h, dx, dy, u_l, mu_a,
                                     max_iter=10_000, tol=1e-6, omega=1.5):
    """
    Solve 2D incompressible Reynolds equation on a rectangular grid:

        ∂/∂x( c ∂p/∂x ) + ∂/∂y( c ∂p/∂y ) = - (u_l / 2) ∂h/∂x

    with Dirichlet BC p = 0 on all boundaries.

    Parameters
    ----------
    h : ndarray (Ny, Nx)
        Film thickness h(x, y) on a uniform grid.
    dx, dy : float
        Grid spacings in x and y.
    u_l : float
        Sliding speed of the lower surface in +x direction.
    mu_a : float
        Dynamic viscosity.
    max_iter : int
        Maximum number of Gauss–Seidel iterations.
    tol : float
        Stopping tolerance on L2 norm of residual.
    omega : float
        Over-relaxation parameter (1 = pure Gauss–Seidel, 1<ω<2 = SOR).

    Returns
    -------
    p : ndarray (Ny, Nx)
        Pressure field p(x, y).
    """
    Ny, Nx = h.shape
    p = np.zeros_like(h)  # initial guess

    # c(x,y) = h^3 / (12 μ)
    c = h**3 / (12 * mu_a)

    for it in range(max_iter):
        max_res = 0.0

        # Loop over interior points (Dirichlet BC: p=0 on boundary)
        for j in range(1, Ny - 1):        # y-index
            for i in range(1, Nx - 1):    # x-index
                # Coefficients at half-nodes (arithmetic averages)
                c_imh = 0.5 * (c[j, i-1] + c[j, i])
                c_iph = 0.5 * (c[j, i]   + c[j, i+1])
                c_jmh = 0.5 * (c[j-1, i] + c[j, i])
                c_jph = 0.5 * (c[j, i]   + c[j+1, i])

                # RHS term: - (u_l / 2) ∂h/∂x  (central difference)
                dhdx = (h[j, i+1] - h[j, i-1]) / (2 * dx)
                rhs = (u_l / 2.0) * dhdx

                # Current neighbor pressures
                p_im = p[j, i-1]
                p_ip = p[j, i+1]
                p_jm = p[j-1, i]
                p_jp = p[j+1, i]

                # Coefficients
                Ax = (c_iph + c_imh) / dx**2
                Ay = (c_jph + c_jmh) / dy**2
                A_center = Ax + Ay

                # Discrete equation:
                # c_iph*(p_ip - p_ij) - c_imh*(p_ij - p_im)   / dx^2
                # + c_jph*(p_jp - p_ij) - c_jmh*(p_ij - p_jm) / dy^2 = rhs
                #
                # → p_ij_new = ( c_iph/dx^2 * p_ip + c_imh/dx^2 * p_im
                #               + c_jph/dy^2 * p_jp + c_jmh/dy^2 * p_jm
                #               - rhs ) / (Ax + Ay)
                p_new = (
                    c_iph * p_ip / dx**2
                    + c_imh * p_im / dx**2
                    + c_jph * p_jp / dy**2
                    + c_jmh * p_jm / dy**2
                    - rhs
                ) / A_center

                # SOR update
                p_old = p[j, i]
                p[j, i] = (1 - omega) * p_old + omega * p_new

                res = abs(p[j, i] - p_old)
                if res > max_res:
                    max_res = res

        if max_res < tol:
            # print(f"Converged in {it+1} iterations, max_res = {max_res:.2e}")
            break

    return p


# ---------------------------------------------------------------------
# Main: build a simple 2D case and compare to 1D analytical solution
# ---------------------------------------------------------------------
def main():
    # --- physical parameters (same scale as Almqvist examples) ---
    k = 1.2          # inclination parameter
    u_l = 5.0        # m/s
    mu_a = 0.01      # Pa·s
    h_T = 10e-6      # m (trailing edge)
    Lx = 0.1         # m (length)
    Ly = 1e-10        # m (width of the 2D domain, arbitrary)

    # --- 2D grid ---
    Nx = 81
    Ny = 41
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # --- film thickness h(x, y) = h(x) from slider geometry ---
    h_line = slider_geometry_h(x, k, h_T, Lx)   # 1D profile
    h = np.tile(h_line, (Ny, 1))                # copy along y

    # --- solve 2D Reynolds ---
    p_2d = solve_reynolds_2d_incompressible(h, dx, dy, u_l, mu_a,
                                            max_iter=5000, tol=1e-6, omega=1.6)

    # --- analytical 1D solution ---
    p_exact_1d = analytic_pressure_linear_slider(x, k, mu_a, u_l, h_T, Lx)

    # --- extract numerical mid-line (y = Ly/2) ---
    j_mid = Ny // 2
    p_mid = p_2d[j_mid, :]

    # --- compare (mid-line vs analytic) ---
    rel_err = np.linalg.norm(p_mid - p_exact_1d) / np.linalg.norm(p_exact_1d)
    print(f"Relative L2 error along mid-line: {rel_err:.3e}")

    # --- plots ---

    # 1) Film thickness (should look like Fig. 5.4 shape)
    plt.figure(figsize=(6, 4))
    plt.plot(x / Lx, h_line * 1e6)
    plt.xlabel("x / Lx")
    plt.ylabel("h(x) [µm]")
    plt.title("Film thickness profile (linear slider)")
    plt.grid(True)
    plt.tight_layout()

    # 2) 2D pressure contour
    plt.figure(figsize=(6, 4))
    cp = plt.contourf(X / Lx, Y / Ly, p_2d, levels=30)
    plt.colorbar(cp, label="p(x, y) [Pa]")
    plt.xlabel("x / Lx")
    plt.ylabel("y / Ly")
    plt.title("2D pressure field from Reynolds equation")
    plt.tight_layout()

    # 3) Mid-line vs analytical
    plt.figure(figsize=(6, 4))
    plt.plot(x / Lx, p_exact_1d, label="Analytical (1D, Eq. 5.106)")
    plt.plot(x / Lx, p_mid, "--", label="2D numerical mid-line")
    plt.xlabel("x / Lx")
    plt.ylabel("p(x) [Pa]")
    plt.title("Mid-line pressure: 2D vs analytical 1D")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

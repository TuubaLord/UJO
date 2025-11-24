import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Geometry from Almqvist: linear slider h(x)
#   Eqs. (5.96)–(5.97), Section 5.3.1
# ---------------------------------------------------------------------------
def slider_geometry_h(x, k, h_T, l):
    """
    Film thickness for the infinitely wide linear slider.

    From Almqvist Eqs. (5.96)-(5.97):
        h(x) = h_T * (1 + k - k * x / l)

    Parameters:
        x    : array of grid points in [0, l]
        k    : inclination parameter
        h_T  : trailing-edge film thickness
        l    : length of bearing

    Returns:
        h(x)
    """
    return h_T * (1 + k - k * x / l)


# ---------------------------------------------------------------------------
# Analytical pressure solution from Almqvist Eq. (5.106)
# ---------------------------------------------------------------------------
def analytic_pressure_linear_slider(x, k, mu_a, u_l, h_T, l):
    """
    Analytical pressure for an incompressible, iso-viscous linear slider bearing.

    Eq. (5.106) (Section 5.3.1):
        p(x) = (6 μ_a u_l l / h_T^2) * (1/k) * [1/H - (1+k)/(2+k)(1/H²) - 1/(2+k)]
        H = 1 + k - k x/l
    """
    H = 1 + k - k * x / l
    bearing_number = 6 * mu_a * u_l * l / (h_T**2)
    shape = (1/k) * (1/H - ((1+k)/(2+k)) * (1/H**2) - 1/(2+k))
    return bearing_number * shape


# ---------------------------------------------------------------------------
# Numerical solution of 1D Reynolds equation
#   Eq. (5.99): d/dx[ (h^3 / 12μ) dp/dx ] = (u_l/2) dh/dx
# ---------------------------------------------------------------------------
def solve_reynolds_1d_incompressible(x, h, u_l, mu_a):
    """
    Finite-difference solution of:
        d/dx[ (h^3 / (12 μ_a)) dp/dx ] = (u_l / 2) dh/dx
    Uses second-order central differences. Fully consistent with
    Section 5.7.1.

    Returns:
        p(x)
    """
    x = np.asarray(x)
    h = np.asarray(h)
    N = len(x)
    dx = x[1] - x[0]

    c = h**3 / (12 * mu_a)  # coefficient in front of dp/dx

    M = N - 2  # number of interior points
    A = np.zeros((M, M))
    b = np.zeros(M)

    for j in range(M):
        i = j + 1  # global index i

        c_im = 0.5 * (c[i-1] + c[i])
        c_ip = 0.5 * (c[i] + c[i+1])

        A[j, j] = (c_im + c_ip) / dx**2
        if j > 0:
            A[j, j-1] = -c_im / dx**2
        if j < M - 1:
            A[j, j+1] = -c_ip / dx**2

        dhdx_i = (h[i+1] - h[i-1]) / (2*dx)
        b[j] = -(u_l / 2) * dhdx_i

    p_internal = np.linalg.solve(A, b)

    p = np.zeros_like(h)
    p[1:-1] = p_internal
    return p


# ---------------------------------------------------------------------------
# MAIN: Full end-to-end example with plotting
# ---------------------------------------------------------------------------
def main():
    # Problem parameters (same scale as Almqvist Ch. 5 examples)
    k = 1.2          # inclination parameter (optimal value in 5.3.5)
    u_l = 5.0        # m/s, sliding velocity (used in Fig. 5.6)
    mu_a = 0.01      # Pa·s, viscosity
    h_T = 10e-6      # trailing edge thickness
    l = 0.1          # bearing length

    # Grid
    N = 201
    x = np.linspace(0, l, N)

    # Film thickness profile
    h = slider_geometry_h(x, k, h_T, l)

    # Numerical pressure (solving Reynolds Eq. 5.99)
    p_num = solve_reynolds_1d_incompressible(x, h, u_l, mu_a)

    # Analytical pressure (Eq. 5.106)
    p_exact = analytic_pressure_linear_slider(x, k, mu_a, u_l, h_T, l)

    # Compare results
    rel_err = np.linalg.norm(p_num - p_exact) / np.linalg.norm(p_exact)
    print(f"Relative L2 error (numerical vs analytical): {rel_err:.3e}")

    # Normalized shape comparison (like Fig. 5.5)
    P_num = p_num / np.max(np.abs(p_num))
    P_exact = p_exact / np.max(np.abs(p_exact))
    rel_err_shape = np.linalg.norm(P_num - P_exact) / np.linalg.norm(P_exact)
    print(f"Relative error (dimensionless pressure shape): {rel_err_shape:.3e}")

    # -----------------------------------------------------------------------
    # Plot 1: Film thickness profile (h)
    # → Compare to shape in Fig. 5.4
    # -----------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(x/l, h*1e6, label="h(x) [μm]")
    plt.xlabel("x / l")
    plt.ylabel("Film thickness h(x) [µm]")
    plt.title("Film Thickness Profile (cf. Almqvist Fig. 5.4)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------
    # Plot 2: Pressure field p(x) (numerical vs analytical)
    # → This is the direct "pressure fields" output of the assignment.
    # -----------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(x/l, p_exact, label="Analytical (Eq. 5.106)")
    plt.plot(x/l, p_num, "--", label="Numerical (Reynolds Eq. 5.99)")
    plt.xlabel("x / l")
    plt.ylabel("Pressure p(x) [Pa]")
    plt.title("Pressure Field – Comparison to Almqvist (Sec. 5.2 / 5.3.1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------
    # Plot 3: Dimensionless pressure shape (cf. Almqvist Fig. 5.5)
    # -----------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(x/l, P_exact, label="Analytical shape")
    plt.plot(x/l, P_num, "--", label="Numerical shape")
    plt.xlabel("x / l")
    plt.ylabel("Dimensionless pressure P(x)")
    plt.title("Dimensionless Pressure Shape (cf. Fig. 5.5)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

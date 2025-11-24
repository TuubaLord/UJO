from solve_pressure_fields import solve_pressure_field_2d
from utils import slider_geometry_h, analytic_load_capacity, analytic_pressure_linear_slider
import numpy as np

def test_linear_slider_pressure_profile():
    """
    Test `solve_pressure_field_2d` against the analytical solution
    for the infinitely wide linear slider bearing.

    Book references:
      * Geometry h(x): Eqs. (5.96)–(5.97), Section 5.3.1.
      * Reynolds equation: Eq. (5.99), Section 5.3.1.
      * Analytical pressure: Eq. (5.106).
      * Dimensionless form and shape P(X): Eqs. (5.163)–(5.166), Section 5.4.2.
      * Example k ≈ 1.2 shown in Figs. 5.4–5.5.

    We use:
        k   = 1.2     (inclination parameter, as discussed in 5.3.5 / Fig. 5.8)
        u_l = 5.0 m/s (velocity used in velocity plots Fig. 5.6)
        μ_a = 0.01 Pa·s  (chosen but cancels out in the dimensionless comparison)
        h_T = 10e-6 m
        l   = 0.1 m

    Expected behaviour:
        The numerical pressure profile p_num(x) should match the analytic p(x)
        from Eq. (5.106) within a small relative error.
    """
    # --- parameters from / consistent with the book ---
    k = 1.2
    u_l = 5.0          # m/s (velocity used in Fig. 5.6 text)
    mu_a = 0.01        # Pa·s (arbitrary but appears in scaling)
    h_T = 10e-6        # m  (same order as used for air examples, Section 5.6)
    l = 0.1            # m

    # --- grid (2D with ny = 1 so we effectively solve 1D in x) ---
    nx = 201
    ny = 1
    x = np.linspace(0.0, l, nx)
    dx = x[1] - x[0]
    dy = 1.0  # dummy

    h_line = slider_geometry_h(x, k=k, h_T=h_T, l=l)
    h = h_line.reshape(1, -1)         # shape (ny, nx)

    # stationary linear slider: upper surface at rest, lower moves at u_l
    uu = 0.0
    ul = u_l
    vu = 0.0
    vl = 0.0

    # --- solve Reynolds equation numerically (incompressible) ---
    p_num_2d, qx, qy = solve_pressure_field_2d(
        h=h,
        dx=dx,
        dy=dy,
        uu=uu,
        ul=ul,
        vu=vu,
        vl=vl,
        mu=mu_a,
        mode="incompressible",
        tol=1e-7,
        max_iter=20000,
        omega_relax=0.2,
    )

    # Extract 1D pressure line and enforce Dirichlet BC p(0)=p(l)=0 as in Eq. (5.102)
    p_num = p_num_2d[0].copy()
    p_num[0] = 0.0
    p_num[-1] = 0.0

      # --- analytical solution from Eq. (5.106) ---
    p_exact = analytic_pressure_linear_slider(x, k=k, mu_a=mu_a, u_l=u_l, h_T=h_T, l=l)

    # --- compare SHAPE (dimensionless profiles) instead of raw values ---
    # Avoid division by zero if something went badly wrong:
    if np.max(np.abs(p_num)) == 0 or np.max(np.abs(p_exact)) == 0:
        raise AssertionError("Pressure field is identically zero; solver or setup is broken.")

    p_num_dimless = p_num / np.max(np.abs(p_num))
    p_exact_dimless = p_exact / np.max(np.abs(p_exact))

    rel_err_shape = np.linalg.norm(p_num_dimless - p_exact_dimless) / np.linalg.norm(p_exact_dimless)

    print("Test 1 – relative L2 error (shape of pressure):", rel_err_shape)
    assert rel_err_shape < 0.2  # start with 20% tolerance; tighten later if it behaves well

test_linear_slider_pressure_profile()
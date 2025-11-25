import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

from reynolds_1d_analytic import slider_geometry_h, analytic_pressure_linear_slider
from reynolds_1d_numeric import solve_reynolds_1d_incompressible
from reynolds_2d_numeric import solve_reynolds_2d_incompressible
from reynolds_2d_journal import solve_reynolds_2d_journal  # currently unused but kept for later use

@st.cache_data(show_spinner=False)
def compute_1d_pressure_fields(k, u_l, mu_a, h_T, Lx, Nx):
    x = np.linspace(0.0, Lx, Nx)
    h_1d = slider_geometry_h(x, k=k, h_T=h_T, Lx=Lx)

    p_analytic = analytic_pressure_linear_slider(x, k, mu_a, u_l, h_T, Lx)
    p_1d_num = solve_reynolds_1d_incompressible(x, h_1d, u_l, mu_a)

    return x, h_1d, p_analytic, p_1d_num


@st.cache_data(show_spinner=False)
def compute_2d_pressure_field(k, u_l, mu_a, h_T, Lx, Ly, Nx, Ny, bc_y):
    x, h_1d, _, _ = compute_1d_pressure_fields(k, u_l, mu_a, h_T, Lx, Nx)
    dx = x[1] - x[0]

    y = np.linspace(0.0, Ly, Ny)
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    h_2d = np.tile(h_1d, (Ny, 1))

    p_2d = solve_reynolds_2d_incompressible(
        h_2d, dx, dy, u_l, mu_a,
        max_iter=4000, tol=1e-5, omega=1.6, bc_y=bc_y
    )

    j_mid = Ny // 2
    p_mid = p_2d[j_mid, :]

    return X, Y, p_2d, p_mid


st.title("Pressure fields for sliding bearing")
with st.sidebar:
    st.header("Parameters")

    # (Currently only incompressible is implemented in this app)
    st.caption("Fluid model: incompressible")

    k = st.slider("Inclination k", 0.1, 3.0, 1.2, 0.05)
    u_l = st.slider("Sliding speed u_l [m/s]", 0.1, 50.0, 5.0, 0.1)

    mu_mpas = st.slider("Viscosity μ [mPa·s]", 0.1, 100.0, 10.0, 0.1)
    mu_a = mu_mpas * 1e-3  # Pa·s

    h_T_um = st.slider("Trailing edge thickness h_T [µm]", 1.0, 50.0, 10.0, 0.5)
    h_T = h_T_um * 1e-6  # m

    Lx_cm = st.slider("Length Lx [cm]", 1.0, 20.0, 10.0, 0.5)
    Lx = Lx_cm / 100.0  # m

    Ly_cm = st.slider("Width Ly [cm] (2D only)", 1.0, 20.0, 5.0, 0.5)
    Ly = Ly_cm / 100.0  # m

    rotation_deg = st.slider(
        "Journal visualization rotation (°)",
        -180.0, 180.0, 0.0, 5.0
    )

    Nx = st.slider("Nx (grid points in x)", 41, 161, 81, 10)
    Ny = st.slider("Ny (grid points in y, 2D)", 3, 81, 41, 2)

    bc_y_mode = st.radio(
        "BC in y-direction",
        ["Neumann (infinite width)", "Dirichlet (p=0 at edges)"],
    )
    bc_y = "neumann" if "Neumann" in bc_y_mode else "dirichlet"

    show_2d = st.checkbox("Compute 2D field", value=True)

# ------------------------
# 1D grid and film thickness
# ------------------------
x, h_1d, p_analytic, p_1d_num = compute_1d_pressure_fields(k, u_l, mu_a, h_T, Lx, Nx)

# ------------------------
# 2D solution (if enabled)
# ------------------------
p_2d = None
p_mid = None
X = Y = None

if show_2d:
    X, Y, p_2d, p_mid = compute_2d_pressure_field(
        k, u_l, mu_a, h_T, Lx, Ly, Nx, Ny, bc_y
    )

# ------------------------
# Plot film thickness
# ------------------------
st.subheader("Film thickness profile")
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(x / Lx, h_1d * 1e6)
ax1.set_xlabel("x / Lx")
ax1.set_ylabel("h(x) [µm]")
ax1.grid(True)
st.pyplot(fig1)

# ------------------------
# Plot pressure curves
# ------------------------
st.subheader("Pressure profiles (1D & 2D mid-line)")
fig2, ax2 = plt.subplots(figsize=(6, 4))

ax2.plot(x / Lx, p_analytic, label="Analytical 1D (incompressible)")

p1_plot = p_1d_num
p_mid_plot = p_mid if p_mid is not None else None
ylabel = "p(x) [Pa]"

ax2.plot(x / Lx, p1_plot, "--", label="1D numerical (incompressible)")

if p_mid_plot is not None:
    ax2.plot(x / Lx, p_mid_plot, ":", label="2D mid-line (incompressible)")

ax2.set_xlabel("x / Lx")
ax2.set_ylabel(ylabel)
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# ------------------------
# 2D contour plot
# ------------------------
if show_2d and p_2d is not None:
    st.subheader("2D pressure field (incompressible)")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    cf = ax3.contourf(X / Lx, Y / Ly, p_2d, levels=40)
    fig3.colorbar(cf, ax=ax3, label="p(x, y) [Pa]")
    ax3.set_xlabel("x / Lx")
    ax3.set_ylabel("y / Ly")
    ax3.set_title(f"2D pressure field (BC_y = {bc_y})")
    st.pyplot(fig3)

# ------------------------
# Error metrics (incompressible)
# ------------------------
st.markdown("### Error metrics vs analytical 1D (incompressible)")
rel_err_1d = np.linalg.norm(p_1d_num - p_analytic) / np.linalg.norm(p_analytic)
st.write(f"Relative L2 error (1D FD vs analytic): `{rel_err_1d:.3e}`")

if p_mid is not None:
    rel_err_mid = np.linalg.norm(p_mid - p_analytic) / np.linalg.norm(p_analytic)
    st.write(f"Relative L2 error (2D mid-line vs analytic): `{rel_err_mid:.3e}`")


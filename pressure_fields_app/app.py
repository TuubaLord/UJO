# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from reynolds_1d_analytic import slider_geometry_h, analytic_pressure_linear_slider
from reynolds_1d_numeric import solve_reynolds_1d_incompressible
from reynolds_2d_numeric import solve_reynolds_2d_incompressible

st.set_page_config(page_title="Reynolds Slider Bearing Demo", layout="wide")
st.title("Reynolds Equation – Linear Slider Bearing Demo")

with st.sidebar:
    st.header("Parameters")

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

    Nx = st.slider("Nx (grid points in x)", 41, 161, 81, 10)
    Ny = st.slider("Ny (grid points in y, 2D)", 3, 81, 41, 2)

    bc_y_mode = st.radio("BC in y-direction", ["Neumann (infinite width)", "Dirichlet (p=0 at edges)"])
    bc_y = "neumann" if "Neumann" in bc_y_mode else "dirichlet"

    show_2d = st.checkbox("Compute 2D field", value=True)

# 1D grid and film thickness
x = np.linspace(0.0, Lx, Nx)
dx = x[1] - x[0]
h_1d = slider_geometry_h(x, k=k, h_T=h_T, Lx=Lx)

# 1D solutions
p_analytic = analytic_pressure_linear_slider(x, k, mu_a, u_l, h_T, Lx)
p_1d_num = solve_reynolds_1d_incompressible(x, h_1d, u_l, mu_a)

# 2D solution (if enabled)
p_2d = None
p_mid = None
X = Y = None

if show_2d:
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

# --- Plot film thickness ---
st.subheader("Film thickness profile")
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(x / Lx, h_1d * 1e6)
ax1.set_xlabel("x / Lx")
ax1.set_ylabel("h(x) [µm]")
ax1.grid(True)
st.pyplot(fig1)

# --- Plot pressure curves ---
st.subheader("Pressure profiles (1D & 2D mid-line)")
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(x / Lx, p_analytic, label="Analytical 1D (Eq. 5.106)")
ax2.plot(x / Lx, p_1d_num, "--", label="1D FD solution")

if p_mid is not None:
    ax2.plot(x / Lx, p_mid, ":", label="2D mid-line")

ax2.set_xlabel("x / Lx")
ax2.set_ylabel("p(x) [Pa]")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# --- 2D contour plot ---
if show_2d and p_2d is not None:
    st.subheader("2D pressure field")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    cf = ax3.contourf(X / Lx, Y / Ly, p_2d, levels=40)
    fig3.colorbar(cf, ax=ax3, label="p(x, y) [Pa]")
    ax3.set_xlabel("x / Lx")
    ax3.set_ylabel("y / Ly")
    ax3.set_title(f"2D pressure field (BC_y = {bc_y})")
    st.pyplot(fig3)

# --- Error metrics ---
st.markdown("### Error metrics vs analytical 1D")
rel_err_1d = np.linalg.norm(p_1d_num - p_analytic) / np.linalg.norm(p_analytic)
st.write(f"Relative L2 error (1D FD vs analytic): `{rel_err_1d:.3e}`")

if p_mid is not None:
    rel_err_mid = np.linalg.norm(p_mid - p_analytic) / np.linalg.norm(p_analytic)
    st.write(f"Relative L2 error (2D mid-line vs analytic): `{rel_err_mid:.3e}`")

# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from reynolds_1d_analytic import slider_geometry_h, analytic_pressure_linear_slider
from reynolds_1d_numeric import (
    solve_reynolds_1d_incompressible,
    solve_reynolds_1d_compressible,   # <-- you implement this
)
from reynolds_2d_numeric import (
    solve_reynolds_2d_incompressible,
    solve_reynolds_2d_compressible,   # <-- and this
)

st.set_page_config(page_title="Reynolds Slider Bearing Demo", layout="wide")
st.title("Reynolds Equation – Linear Slider Bearing Demo")

with st.sidebar:
    st.header("Parameters")

    # Flow model selection
    flow_model = st.radio(
        "Fluid model",
        ["Incompressible", "Compressible (ideal gas)"],
        index=0,
    )

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

    if flow_model.startswith("Compressible"):
        st.markdown("---")
        st.caption("Compressible model parameters")
        p_ref = st.slider("Reference pressure p_ref [bar]", 0.1, 10.0, 1.0, 0.1)
        p_ref_pa = p_ref * 1e5  # bar → Pa
        rho_ref = st.slider("Reference density ρ_ref [kg/m³]", 0.1, 50.0, 10.0, 0.1)
    else:
        p_ref_pa = None
        rho_ref = None

# 1D grid and film thickness
x = np.linspace(0.0, Lx, Nx)
dx = x[1] - x[0]
h_1d = slider_geometry_h(x, k=k, h_T=h_T, Lx=Lx)

# --- 1D solution depending on model ---
if flow_model == "Incompressible":
    p_analytic = analytic_pressure_linear_slider(x, k, mu_a, u_l, h_T, Lx)
    p_1d_num = solve_reynolds_1d_incompressible(x, h_1d, u_l, mu_a)
else:
    # Compressible: no closed-form analytic solution here
    p_analytic = None
    p_1d_num = solve_reynolds_1d_compressible(
        x=x,
        h=h_1d,
        u_l=u_l,
        mu_a=mu_a,
        p_ref=p_ref_pa,
        rho_ref=rho_ref,
        max_iter=500,
        tol=1e-6,
    )

# 2D solution (if enabled)
p_2d = None
p_mid = None
X = Y = None

if show_2d:
    y = np.linspace(0.0, Ly, Ny)
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    h_2d = np.tile(h_1d, (Ny, 1))

    if flow_model == "Incompressible":
        p_2d = solve_reynolds_2d_incompressible(
            h_2d, dx, dy, u_l, mu_a,
            max_iter=4000, tol=1e-5, omega=1.6, bc_y=bc_y
        )
    else:
        p_2d = solve_reynolds_2d_compressible(
            h=h_2d,
            dx=dx,
            dy=dy,
            u_l=u_l,
            mu_a=mu_a,
            p_ref=p_ref_pa,
            rho_ref=rho_ref,
            max_iter=2000,
            tol=1e-5,
            omega=1.4,
            bc_y=bc_y,
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

if p_analytic is not None:
    ax2.plot(x / Lx, p_analytic, label="Analytical 1D (incompressible)")

if flow_model.startswith("Compressible"):
    p1_plot = p_1d_num - p_ref_pa
    p_mid_plot = p_mid - p_ref_pa if p_mid is not None else None
    ylabel = "p(x) - p_ref [Pa]"
else:
    p1_plot = p_1d_num
    p_mid_plot = p_mid
    ylabel = "p(x) [Pa]"

ax2.plot(x / Lx, p1_plot, "--", label=f"1D numerical ({flow_model})")

if p_mid_plot is not None:
    ax2.plot(x / Lx, p_mid_plot, ":", label=f"2D mid-line ({flow_model})")

ax2.set_xlabel("x / Lx")
ax2.set_ylabel(ylabel)
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)


# --- 2D contour plot ---
if show_2d and p_2d is not None:
    st.subheader(f"2D pressure field ({flow_model})")
    fig3, ax3 = plt.subplots(figsize=(7, 4))

    if flow_model.startswith("Compressible"):
        field = p_2d - p_ref_pa
        clabel = "p(x, y) - p_ref [Pa]"
    else:
        field = p_2d
        clabel = "p(x, y) [Pa]"

    cf = ax3.contourf(X / Lx, Y / Ly, field, levels=40)
    fig3.colorbar(cf, ax=ax3, label=clabel)
    ax3.set_xlabel("x / Lx")
    ax3.set_ylabel("y / Ly")
    ax3.set_title(f"2D pressure field (BC_y = {bc_y})")
    st.pyplot(fig3)

# --- Error metrics (only meaningful for incompressible) ---
if flow_model == "Incompressible" and p_analytic is not None:
    st.markdown("### Error metrics vs analytical 1D (incompressible)")
    rel_err_1d = np.linalg.norm(p_1d_num - p_analytic) / np.linalg.norm(p_analytic)
    st.write(f"Relative L2 error (1D FD vs analytic): `{rel_err_1d:.3e}`")

    if p_mid is not None:
        rel_err_mid = np.linalg.norm(p_mid - p_analytic) / np.linalg.norm(p_analytic)
        st.write(f"Relative L2 error (2D mid-line vs analytic): `{rel_err_mid:.3e}`")
else:
    st.markdown("### Compressible case")
    st.write("Analytical incompressible solution is hidden in compressible mode.")

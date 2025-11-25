import streamlit as st
import ross as rs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from io import BytesIO
from tempfile import NamedTemporaryFile
from solve_K_C import solve_K_C

# =========================================================
#  Cached builders
# =========================================================
@st.cache_resource(show_spinner=False)
def get_overhung_rotor():
    return build_overhung_rotor()


# =========================================================
#  Build the rotor
# =========================================================
def build_overhung_rotor():
    steel = rs.Material(
        name="steel",
        rho=7810,
        E=211e9,
        Poisson=0.3,
    )

    L_total = 1.5
    n_elems = 15
    L_e = L_total / n_elems

    shaft_elems = []
    for i in range(n_elems):
        shaft_elems.append(
            rs.ShaftElement(
                L=L_e,
                idl=0.0,
                odl=0.025,
                idr=0.0,
                odr=0.025,
                material=steel
            )
        )

    # Disk at node n_elems
    r_disk = 0.125
    t_disk = 0.04
    rho = 7810

    m_disk = rho * np.pi * r_disk**2 * t_disk
    Id_disk = (1/12) * m_disk * (3*r_disk**2 + t_disk**2)
    Ip_disk = 0.5 * m_disk * r_disk**2

    disk = rs.DiskElement(
        n=n_elems,
        m=m_disk,
        Id=Id_disk,
        Ip=Ip_disk
    )

    rotor = rs.Rotor(
        shaft_elements=shaft_elems,
        disk_elements=[disk],
        bearing_elements=[]
    )

    return rotor


# =========================================================
#  Compute eigenvalues at spin speed Omega
# =========================================================
def compute_eigenvalues(rotor, Kb_dict, Cb_dict, Omega):

    M = rotor.M()
    K = rotor.K(Omega)
    C = rotor.C(Omega)

    try:
        G = rotor.G(Omega)
    except TypeError:
        G = rotor.G()

    # Insert bearings
    K = K.copy()
    C = C.copy()

    for node, Kb in Kb_dict.items():
        d = 6 * node
        K[d:d+2, d:d+2] += Kb_dict[node]
        C[d:d+2, d:d+2] += Cb_dict[node]

    # Assemble state-space
    n = M.shape[0]
    Z = np.zeros_like(M)
    I = np.eye(n)

    A = np.block([
        [Z, I],
        [-np.linalg.solve(M, K), -np.linalg.solve(M, C + Omega*G)]
    ])

    eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs


# =========================================================
#  BENDING PARTICIPATION SCORE
# =========================================================
def bending_score(mode_shape, rotor):
    """
    Measures how much the mode uses bending DOFs: x, y, alpha, beta
    """
    score = 0.0
    n_nodes = len(rotor.nodes)

    for node in range(n_nodes):
        base = 6 * node
        x = mode_shape[base]
        y = mode_shape[base + 1]
        alpha = mode_shape[base + 3]
        beta = mode_shape[base + 4]

        score += np.abs(x)**2 + np.abs(y)**2 + np.abs(alpha)**2 + np.abs(beta)**2

    return score


# =========================================================
#  CAMPBELL — BENDING MODES ONLY
# =========================================================
def campbell_bending(rotor, Kb_dict, Cb_dict, speeds, n_modes_plot=5):

    mode_freqs = []

    for idx, Omega in enumerate(speeds):

        eigvals, eigvecs = compute_eigenvalues(rotor, Kb_dict[idx], Cb_dict[idx], Omega)

        freqs = np.abs(np.imag(eigvals)) / (2*np.pi)  # Hz

        # Compute bending score for each mode
        scores = np.array([bending_score(eigvecs[:, i], rotor) for i in range(len(eigvals))])

        # Select top bending modes
        idx = np.argsort(-scores)[:n_modes_plot]

        mode_freqs.append(freqs[idx])

    return np.array(mode_freqs)


def MAC(phi, psi):
    phi = phi.astype(complex)
    psi = psi.astype(complex)
    num = np.abs(np.vdot(phi, psi))**2
    den = (np.vdot(phi, phi) * np.vdot(psi, psi))
    return num / den


def extract_bending_modes(rotor, eigvals, eigvecs, n_modes=12):

    scores = np.array([bending_score(eigvecs[:, i], rotor)
                       for i in range(len(eigvals))])

    idx = np.argsort(-scores)[:n_modes]
    vals = eigvals[idx]
    vecs = eigvecs[:, idx]

    return vals, vecs


def modal_tracking(rotor, Kb_dict, Cb_dict, speeds, n_modes=6):
    """
    Returns:
        tracked_freqs: (n_speeds × n_modes)
        tracked_eigvals: list of eigenvalue arrays
        tracked_modes: list of mode shape matrices
    """

    tracked_freqs = np.zeros((len(speeds), n_modes))
    tracked_eigvals = []
    tracked_modes = []

    # ---- Compute modes at first speed ----
    eigvals, eigvecs = compute_eigenvalues(rotor, Kb_dict[0], Cb_dict[0], speeds[0])
    vals, vecs = extract_bending_modes(rotor, eigvals, eigvecs, n_modes)

    tracked_eigvals.append(vals)
    tracked_modes.append(vecs)

    tracked_freqs[0, :] = np.abs(np.imag(vals)) / (2*np.pi)

    # ---- Track through remaining speeds ----
    for k in range(1, len(speeds)):
        eigvals_new, eigvecs_new = compute_eigenvalues(rotor, Kb_dict[k], Cb_dict[k], speeds[k])

        vals_new, vecs_new = extract_bending_modes(rotor, eigvals_new, eigvecs_new, n_modes)

        MAC_matrix = np.zeros((n_modes, n_modes))

        for i in range(n_modes):
            for j in range(n_modes):
                MAC_matrix[i, j] = MAC(tracked_modes[-1][:, i], vecs_new[:, j])

        assignment = np.argmax(MAC_matrix, axis=1)

        matched_vals = vals_new[assignment].astype(complex)
        matched_vecs = vecs_new[:, assignment].astype(complex)

        tracked_eigvals.append(matched_vals)
        tracked_modes.append(matched_vecs)

        tracked_freqs[k, :] = np.abs(np.imag(matched_vals)) / (2*np.pi)

    return tracked_freqs, tracked_eigvals, tracked_modes


# =========================================================
#  CAMPBELL PLOT
# =========================================================
def plot_campbell(speeds_rpm, freqs, eigvals_list, modes_list):
    """
    Generates a Campbell plot with mode tracking and whirl direction.
    Returns the Matplotlib figure so Streamlit can render it.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    Omega_1x = speeds_rpm / 60.0  # 1× running speed in Hz



    for mode in range(freqs.shape[1]):
        mode_freqs = freqs[:, mode]

        whirl = np.array([
            "FW" if np.imag(eigvals_list[i][mode]) > 0 else "BW"
            for i in range(len(speeds_rpm))
        ])

        is_FW = whirl == "FW"
        is_BW = whirl == "BW"
        if mode == 0:
            ax.plot(
                speeds_rpm[is_FW], mode_freqs[is_FW],
                'b-', lw=2,
                label=f"Forward whirl"
            )
            ax.plot(
                speeds_rpm[is_BW], mode_freqs[is_BW],
                'r--', lw=2,
                label=f"Backward whirl", zorder=10
        )
        else:
            ax.plot(
                speeds_rpm[is_FW], mode_freqs[is_FW],
                'b-', lw=2,
                #label=f"Mode {mode + 1} FW"
            )
            ax.plot(
                speeds_rpm[is_BW], mode_freqs[is_BW],
                'r--', lw=2,
                zorder=10
            )

    ax.plot(speeds_rpm, Omega_1x, '--k', linewidth=2, label="1× Running Speed")

    ax.set_xlabel("Speed (RPM)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Campbell Diagram with Automatic Modal Tracking (MAC)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return fig


# =========================================================
#  Cached numerical routines
# =========================================================
@st.cache_data(show_spinner=False)
def get_bearing_matrices(example_book, speeds_rad, D, eta, L, f, c):
    if example_book:
        Kb_dict = {
            0: np.array([[12.81, 16.39], [-25.06, 8.815]]) * 1e6,
            10: np.array([[12.81, 16.39], [-25.06, 8.815]]) * 1e6
        }

        Cb_dict = {
            0: np.array([[232.9, -81.92], [-81.92, 294.9]]) * 1e3,
            10: np.array([[232.9, -81.92], [-81.92, 294.9]]) * 1e3
        }

        kbs = [Kb_dict for _ in range(len(speeds_rad))]
        cbs = [Cb_dict for _ in range(len(speeds_rad))]
        return kbs, cbs

    kbs = []
    cbs = []
    for speed in speeds_rad:
        K, C, _ = solve_K_C(D, speed, eta, L, f, c)

        kbs.append({0: K.astype(np.float64), 10: K.astype(np.float64)})
        cbs.append({0: C.astype(np.float64), 10: C.astype(np.float64)})

    return kbs, cbs


@st.cache_data(show_spinner=False, hash_funcs={np.ndarray: lambda a: a.tobytes()})
def get_modal_results(speeds_rad, kbs, cbs):
    rotor = get_overhung_rotor()

    freqs = campbell_bending(rotor, kbs, cbs, speeds_rad, n_modes_plot=12)

    tracked_freqs, eigvals_list, modes_list = modal_tracking(
        rotor, kbs, cbs, speeds_rad, n_modes=12
    )

    return freqs, eigvals_list, modes_list


# =========================================================
#  ANIMATION OF MODE SHAPE
# =========================================================
def animate_mode_shape(rotor, modes_list, eigvals_list, L_total, n_elems, mode_idx=0):
    """
    Create a GIF animation of a bending mode shape and return it as BytesIO
    so Streamlit can display it with st.image.

    Includes:
    - FW/BW labeling
    - Natural frequency information
    - Dashed horizontal line
    """
    # ----------------------------
    #  Extract mode shape
    # ----------------------------
    mode_shape = modes_list[-1][:n_elems * 6, mode_idx]
    x_dofs = mode_shape[0::6].astype(complex)
    x_dofs_norm = x_dofs / np.max(np.abs(x_dofs))

    shaft_x = np.linspace(0, L_total, n_elems)

    # ----------------------------
    #  Natural frequency (last speed)
    # ----------------------------
    eigvals_last = eigvals_list[-1]
    eig = eigvals_last[mode_idx]
    nat_freq = np.abs(np.imag(eig)) / (2 * np.pi)  # Hz

    # ----------------------------
    #  Create figure
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, L_total)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Axial position (m)")
    ax.set_ylabel("Normalized displacement")

    # Title including FW/BW and natural frequency
    title = (
        f"Mode shape animation (Mode {mode_idx//2 + 1} "
        f"{'(FW)' if mode_idx % 2 == 0 else '(BW)'})\n"
        f"Natural frequency: {nat_freq:.2f} Hz"
    )
    ax.set_title(title)

    ax.grid(True)
    ax.axhline(0, linestyle="--", linewidth=1)  # dashed neutral axis

    line, = ax.plot([], [], "-o")

    # ----------------------------
    #  Animation function
    # ----------------------------
    n_frames = 60

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        phase = 2 * np.pi * frame / n_frames
        disp = np.real(x_dofs_norm * np.exp(1j * phase))
        line.set_data(shaft_x, disp)
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        blit=True,
        interval=100
    )

    # ----------------------------
    #  Save GIF to a temp file
    # ----------------------------
    with NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        temp_path = tmp.name

    writer = PillowWriter(fps=20)
    ani.save(temp_path, writer=writer)
    plt.close(fig)

    # ----------------------------
    #  Load GIF bytes into memory
    # ----------------------------
    with open(temp_path, "rb") as f:
        gif_bytes = BytesIO(f.read())

    gif_bytes.seek(0)
    return gif_bytes
    
st.title("Ujo roottori 👉👈🥺")

with st.sidebar:
    example_book = st.checkbox("Calculate with Friswell example 5.5", value=False)

    D = st.slider("Diamater of bearing [mm]", 10, 150, 100, 10) * 1e-3
    # omega
    eta = st.slider("Oil viscosity [Pa*s]", 0.01, 1.0, 0.1, 0.01)
    L = st.slider("Bearing length (mm)", 10, 50, 30, 10) * 1e-3
    f = st.slider("Static load (N)", 100, 1000, 525, 5)
    c = st.slider("Clearance between shaft and bearing (mm)", 0.01, 0.5, 0.1, 0.01) * 1e-3

rotor = get_overhung_rotor()

# Speed sweep
speeds_rpm = np.linspace(1000, 6000, 40)
speeds_rad = speeds_rpm * 2 * np.pi / 60

# Bearings
kbs, cbs = get_bearing_matrices(example_book, speeds_rad, D, eta, L, f, c)

    # k_xx = 0.2e6
    # k_yy = 0.4e6
    # k_xy = 0.0
    # k_yx = 0.0
    # c_xx = 0.0
    # c_yy = 0.0
    # c_xy = 0.0
    # c_yx = 0.0

    # Kb_dict = {
    #     0: np.array([[k_xx, k_xy], [k_yx, k_yy]]),
    #     10: np.array([[k_xx, k_xy], [k_yx, k_yy]])
    # }

    # Cb_dict = {
    #     0: np.array([[c_xx, c_xy], [c_yx, c_yy]]),
    #     10: np.array([[c_xx, c_xy], [c_yx, c_yy]])
    # }

# Bend-only Campbell + modal tracking (cached)
freqs, eigvals_list, modes_list = get_modal_results(
    speeds_rad, kbs, cbs
)

# ---------------------------------------------------------
# Plot Campbell Diagram
# ---------------------------------------------------------
campbell_fig = plot_campbell(speeds_rpm, freqs, eigvals_list, modes_list)
st.pyplot(campbell_fig)

# ---------------------------------------------------------
# Mode shape animation with selectable mode
# ---------------------------------------------------------
st.subheader("Mode shape animation")

# infer how many modes we have from modes_list
n_modes = modes_list[-1].shape[1]


mode_idx = st.selectbox(
    "Select mode to animate:",
    options=list(range(n_modes)),
    format_func=lambda i: f"Mode {i//2 + 1} ({'FW' if i % 2 == 0 else 'BW'})"
)


gif_bytes = animate_mode_shape(
    rotor,
    modes_list,
    eigvals_list,   
    L_total=1.5,
    n_elems=15,
    mode_idx=mode_idx
)

st.image(
    gif_bytes,
    caption=f"Mode {mode_idx//2 + 1} ({'FW' if mode_idx % 2 == 0 else 'BW'}) bending animation"
)

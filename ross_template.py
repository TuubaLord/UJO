import ross as rs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

    for Omega in speeds:

        eigvals, eigvecs = compute_eigenvalues(rotor, Kb_dict, Cb_dict, Omega)

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
    eigvals, eigvecs = compute_eigenvalues(rotor, Kb_dict, Cb_dict, speeds[0])
    vals, vecs = extract_bending_modes(rotor, eigvals, eigvecs, n_modes)

    tracked_eigvals.append(vals)
    tracked_modes.append(vecs)

    tracked_freqs[0, :] = np.abs(np.imag(vals)) / (2*np.pi)

    # ---- Track through remaining speeds ----
    for k in range(1, len(speeds)):
        eigvals_new, eigvecs_new = compute_eigenvalues(rotor, Kb_dict, Cb_dict, speeds[k])

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
#  MAIN
# =========================================================
if __name__ == "__main__":

    rotor = build_overhung_rotor()

    # Bearings
    k_xx = 0.2e6
    k_yy = 0.4e6
    k_xy = 0.0
    k_yx = 0.0
    c_xx = 0.0
    c_yy = 0.0
    c_xy = 0.0
    c_yx = 0.0

    Kb_dict = {
        0: np.array([[k_xx, k_xy], [k_yx, k_yy]]),
        10: np.array([[k_xx, k_xy], [k_yx, k_yy]])
    }

    Cb_dict = {
        0: np.array([[c_xx, c_xy], [c_yx, c_yy]]),
        10: np.array([[c_xx, c_xy], [c_yx, c_yy]])
    }

    # Speed sweep
    speeds_rpm = np.linspace(0, 6000, 40)
    speeds_rad = speeds_rpm * 2 * np.pi / 60

    # Bend-only Campbell
    freqs = campbell_bending(rotor, Kb_dict, Cb_dict, speeds_rad, n_modes_plot=12)
    print("Campbell bending freq matrix shape:", freqs.shape)

    # ---------------------------------------------------------
    # Plot Campbell with whirl direction + critical crossings
    # ---------------------------------------------------------
    # ---- Run the modal tracking ----
    freqs, eigvals_list, modes_list = modal_tracking(
        rotor, Kb_dict, Cb_dict, speeds_rad, n_modes=12
    )

    # ---- Plot Campbell Plot ----
    plt.figure(figsize=(10, 6))

    Omega_1x = speeds_rpm / 60.0  # 1× running speed in Hz

    for mode in range(freqs.shape[1]):

        mode_freqs = freqs[:, mode]

        # classify whirl direction based on imaginary part of eigenvalues
        whirl = np.array([
            "FW" if np.imag(eigvals_list[i][mode]) > 0 else "BW"
            for i in range(len(speeds_rpm))
        ])

        # FW / BW masking
        is_FW = whirl == "FW"
        is_BW = whirl == "BW"

        plt.plot(
            speeds_rpm[is_FW], mode_freqs[is_FW],
            'b-', lw=2,
            label=f"Mode {mode + 1} FW"
        )
        plt.plot(
            speeds_rpm[is_BW], mode_freqs[is_BW],
            'r--', lw=2,
            label=f"Mode {mode + 1} BW", zorder=10
        )

    # 1× running speed line
    plt.plot(speeds_rpm, Omega_1x, '--k', linewidth=2, label="1× Running Speed")

    plt.xlabel("Speed (RPM)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Campbell Diagram with Automatic Modal Tracking (MAC)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Show Campbell plot and close it after a brief pause
    plt.show(block=False)
    plt.pause(5)  # Pause for 1 second to view the Campbell plot before closing
    plt.close()

    L_total = 1.5
    n_elems = 16

    # ---- Create the animation ----
    # Select the mode for animation (let's use the first mode)
    mode_idx = 0
    mode_shape = modes_list[-1][:n_elems*6, mode_idx]  # Last speed (or any you like)
    mode_shape = mode_shape[::6]  # Extract x-displacements only for simplicity
    # NOTE: To visualize full 3D mode shapes, consider using y-displacements and rotations as well
    # mode_shape_y = mode_shape[1::6]  # Extract y-displacements only for simplicity

    # Plotting for the animation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel("X displacement (m)")
    ax.set_ylabel("Y displacement (m)")
    ax.set_title("Mode Shape Animation")

    # Plot rotor shaft and disk
    shaft_x = np.linspace(0, L_total, n_elems)
    shaft_y = np.zeros_like(shaft_x)
    ax.plot(shaft_x, shaft_y, 'k-', lw=2)  # Shaft line
    ax.plot(L_total, 0, 'bo', markersize=10)  # Disk marker (for simplicity)
    ax.set_xlim([-0.1, L_total + 0.1])
    ax.set_ylim([-1.0, 1.0])

    # Create a line for plotting mode shapes
    line, = ax.plot([], [], 'b-', lw=2)
    theta = np.linspace(0, 2 * np.pi, 100)

    # Animation function
    def update(frame):
        # Get mode displacement at each speed
        displacement = np.abs(mode_shape) * np.cos(theta[frame] + np.angle(mode_shape))  # Adjust for rotation
        # NOTE: To visualize y-displacements instead, uncomment the following line
        # displacement_y = np.abs(mode_shape_y) * np.cos(theta[frame] + np.angle(mode_shape_y))  # Adjust for rotation

        # Update mode shape (displacement) on plot
        line.set_data(shaft_x, displacement/np.max(np.abs(mode_shape)))  # Scale for visibility
        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(theta), interval=100, blit=True)

    # Show animation
    plt.tight_layout()
    plt.show()

import ross as rs
import numpy as np
import matplotlib.pyplot as plt


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
    speeds_rad = speeds_rpm * 2*np.pi/60

    # Bend-only Campbell
    freqs = campbell_bending(rotor, Kb_dict, Cb_dict, speeds_rad, n_modes_plot=12)
    print("Campbell bending freq matrix shape:", freqs.shape)

    # ---------------------------------------------------------
    # Plot Campbell with whirl direction + critical crossings
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    Omega_1x = speeds_rpm / 60.0  # synchronous excitation line (Hz)

    critical_points_rpm = []
    critical_points_hz = []

    for mode_index in range(freqs.shape[1]):

        # At each speed, get the eigenvalue of this mode
        mode_freqs = []
        mode_whirl = []

        for i, Omega in enumerate(speeds_rad):
            eigvals, eigvecs = compute_eigenvalues(rotor, Kb_dict, Cb_dict, Omega)

            # sort modes by bending participation
            scores = np.array([bending_score(eigvecs[:, j], rotor) for j in range(len(eigvals))])
            idx_sorted = np.argsort(-scores)
            mode_id = idx_sorted[mode_index]

            lam = eigvals[mode_id]

            freq_hz = np.abs(np.imag(lam)) / (2*np.pi)
            mode_freqs.append(freq_hz)

            # whirl determination
            if np.real(lam) > 0:
                mode_whirl.append("FW")
            else:
                mode_whirl.append("BW")

        mode_freqs = np.array(mode_freqs)

        # Split forward / backward
        is_FW = np.array([w == "FW" for w in mode_whirl])
        is_BW = ~is_FW

        # Plot FW/BW separately
        plt.plot(speeds_rpm[is_FW], mode_freqs[is_FW],
                'b-', linewidth=2, label=f"Mode {mode_index+1} FW" if mode_index == 0 else "")

        plt.plot(speeds_rpm[is_BW], mode_freqs[is_BW],
                'r-', linewidth=2, label=f"Mode {mode_index+1} BW" if mode_index == 0 else "")

        # detect crossings where |mode_freq - sync| < tolerance
        tol = 0.5  # Hz tolerance for critical speed detection
        for i in range(len(speeds_rpm)):
            if abs(mode_freqs[i] - Omega_1x[i]) < tol and mode_whirl[i] == "FW":
                critical_points_rpm.append(speeds_rpm[i])
                critical_points_hz.append(mode_freqs[i])

    # plot 1× line
    plt.plot(speeds_rpm, Omega_1x, '--k', linewidth=2, label="1× Running Speed")

    # plot critical speed markers
    plt.scatter(critical_points_rpm, critical_points_hz,
                s=80, color='yellow', edgecolors='black', zorder=5, label="Critical Speed")

    plt.xlabel("Speed (RPM)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Campbell Diagram — Bending Modes (FW/BW + Critical Speeds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
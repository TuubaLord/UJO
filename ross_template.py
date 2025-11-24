import ross as rs
import numpy as np

# -----------------------------
# Build the rotor
# -----------------------------
def build_overhung_rotor():
    # Material
    steel = rs.Material(
        name="steel",
        rho=7810,
        E=211e9,
        Poisson=0.3,
    )

    # Shaft
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

    # Disk at overhung end (node n_elems)
    r_disk = 0.125   # radius
    t_disk = 0.04    # thickness
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
        bearing_elements=[]  # bearings will be added manually
    )

    return rotor

# -----------------------------
# Compute eigenvalues with bearing matrices
# -----------------------------
def compute_eigenvalues(rotor, Kb_dict, Cb_dict, Omega):
    """
    rotor : rs.Rotor
    Kb_dict : dict {node : 2x2 bearing stiffness}
    Cb_dict : dict {node : 2x2 bearing damping}
    Omega : float, spin speed
    """

    # Base rotor matrices
    M = rotor.M()
    K = rotor.K(0.0)  # static stiffness
    C = rotor.C(0.0)  # static damping

    # G matrix
    try:
        G = rotor.G(0.0)
    except TypeError:
        G = rotor.G()

    # -----------------------------
    # Insert bearing matrices (x,y DOFs only)
    # -----------------------------
    K = K.copy()
    C = C.copy()

    for node, Kb in Kb_dict.items():
        d_x = 6*node
        d_y = 6*node + 1

        K[d_x:d_x+2, d_x:d_x+2] += Kb_dict[node]
        C[d_x:d_x+2, d_x:d_x+2] += Cb_dict[node]

    # -----------------------------
    # Assemble linearized state-space
    # M qdd + (C + Omega*G) qd + K q = 0
    # -----------------------------
    n = M.shape[0]
    zeros = np.zeros_like(M)
    I = np.eye(n)

    A = np.block([
        [zeros, I],
        [-np.linalg.solve(M, K), -np.linalg.solve(M, C + Omega*G)]
    ])

    eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    rotor = build_overhung_rotor()

    # Example bearing matrices at node 0 and node 10
    k_val = 10e6  # N/m
    c_val = 0.0   # N*s/m

    Kb_dict = {
        0: np.array([[k_val, 0], [0, k_val]]),
        10: np.array([[k_val, 0], [0, k_val]])
    }

    Cb_dict = {
        0: np.array([[c_val, 0], [0, c_val]]),
        10: np.array([[c_val, 0], [0, c_val]])
    }

    # Example rotational speed
    Omega = 0.0  # rad/s

    eigvals, eigvecs = compute_eigenvalues(rotor, Kb_dict, Cb_dict, Omega)

    # Print natural frequencies (Hz)
    wn = np.abs(np.imag(eigvals)) / (2*np.pi)
    print("Natural frequencies (Hz):", np.sort(wn))



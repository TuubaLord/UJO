import numpy as np
import ross as rs
from scipy.integrate import solve_ivp

# =========================================================
# 1. Build Rotor (Overhung) – You can customize everything
# =========================================================
def build_overhung_rotor():
    # --- Shaft elements ---
    shaft = rs.ShaftElement(
        L=0.25,
        material=rs.steel,
        ID=0.02,
        OD=0.04,
        n=0
    )

    # --- Overhung disk ---
    disk = rs.DiskElement.from_geometry(
        n=1,   # node = free end of shaft
        material=rs.steel,
        width=0.03,
        ID=0.02,
        OD=0.15
    )

    # --- Placeholder bearings (will be replaced) ---
    # Only needed to satisfy ROSS building procedure
    bearingL = rs.BearingElement(0, kxx=1e6, kyy=1e6)
    bearingR = rs.BearingElement(1, kxx=1e6, kyy=1e6)

    rotor = rs.Rotor(
        shaft_elements=[shaft],
        disk_elements=[disk],
        bearing_elements=[bearingL, bearingR]
    )

    return rotor


# =========================================================
# 2. Extract Global Matrices from ROSS
# =========================================================
def extract_matrices(rotor, Omega):
    M = rotor.M()          # Mass matrix
    K = rotor.K()          # Structural stiffness
    C = rotor.C()          # Structural damping
    G = rotor.G() * Omega  # Gyroscopic term scaled by speed
    return M, K, C, G


# =========================================================
# 3. Insert Bearing Matrices from Your Fluid-Film Code
# =========================================================
def insert_bearing_matrix(global_matrix, submatrix, node):
    """
    Inserts a 2x2 bearing stiffness/damping block into the global matrix.
    Only x and y translational DOFs are modified.
    Node ordering: [x, y, x', y'] → so DOF index = node*4
    """
    dofs = [node * 4, node * 4 + 1]  # x, y DOFs

    for i in range(2):
        for j in range(2):
            global_matrix[dofs[i], dofs[j]] += submatrix[i, j]


def apply_bearing_matrices(K, C, bearing_data):
    """
    bearing_data = {
        node_index: {
            'K': 2x2 stiffness matrix,
            'C': 2x2 damping matrix
        },
        ...
    }
    """
    for node, mats in bearing_data.items():
        if "K" in mats:
            insert_bearing_matrix(K, mats["K"], node)
        if "C" in mats:
            insert_bearing_matrix(C, mats["C"], node)


# =========================================================
# 4. Equations of Motion (for time-integration)
# =========================================================
def state_space_rhs(t, state, M, C, G, K):
    n = M.shape[0]
    q = state[:n]
    dq = state[n:]

    # solve M * ddq = -(C+G)*dq - K*q
    ddq = np.linalg.solve(M, -(C + G) @ dq - K @ q)

    return np.concatenate((dq, ddq))


# =========================================================
# 5. Full Assembly (Rotor + Bearings + EOM Template)
# =========================================================
def build_full_system(Omega, bearing_data):
    rotor = build_overhung_rotor()
    M, K, C, G = extract_matrices(rotor, Omega)

    # Insert your realistic bearing matrices
    apply_bearing_matrices(K, C, bearing_data)

    return M, K, C, G


# =========================================================
# 6. Example Usage
# =========================================================
if __name__ == "__main__":

    # ---------------------------------------------
    # Example: Bearing matrices from your bearing code
    # Replace these with output from Reynolds solver
    # ---------------------------------------------
    Kb = np.array([[8e6, 1e6],
                   [1e6,  6e6]])

    Cb = np.array([[300, 0],
                   [0, 250]])

    bearing_data = {
        0: {"K": Kb, "C": Cb},   # bearing at node 0
        1: {"K": Kb, "C": Cb},   # bearing at node 1
    }

    # Rotor speed
    Omega = 500 * 2*np.pi/60  # rad/s

    # Build full system
    M, K, C, G = build_full_system(Omega, bearing_data)

    # Initial conditions
    n = M.shape[0]
    q0 = np.zeros(n)
    dq0 = np.zeros(n)
    x0 = np.concatenate((q0, dq0))

    # Integrate 0–1 s
    sol = solve_ivp(
        fun=lambda t, y: state_space_rhs(t, y, M, C, G, K),
        t_span=[0, 1],
        y0=x0,
        max_step=1e-4
    )

    print("Integration successful:", sol.success)

import numpy as np


def solve_reynolds_2d_journal(h, R, dtheta, dz, omega_rotor,
                              mu_a, max_iter=5000, tol=1e-5,
                              omega_relax=1.6, bc_z="neumann"):
    """
    Solve 2D incompressible Reynolds equation for a journal bearing in (θ, z):

        ∂_s ( c ∂_s p ) + ∂_z ( c ∂_z p ) = (u_l / 2) ∂_s h,
        c = h^3 / (12 μ_a),
        s = R θ

    Here we use a sign convention for the rotor:

        omega_rotor > 0  : clockwise rotation
        omega_rotor < 0  : counterclockwise rotation

    The coordinate θ is assumed to increase counterclockwise, so the
    circumferential surface speed used in the Reynolds equation is

        u_l = - R * omega_rotor

    Discretization
    --------------
    Grid indices:
        i = 0..Nθ-1  (circumferential direction, θ)
        j = 0..Nz-1  (axial direction, z)

    Boundary conditions
    -------------------
    θ-direction: periodic
        p(j, 0)     = p(j, Nθ-1)
        p(j, Nθ-1)  = p(j, 0)

    z = 0, z = Lz :
        bc_z = "dirichlet" → p = 0
        bc_z = "neumann"   → ∂p/∂z = 0 (implemented by copying interior rows)

    Parameters
    ----------
    h : ndarray (Nz, Nθ)
        Film thickness h(θ, z) [m].
    R : float
        Journal radius [m].
    dtheta : float
        Grid spacing in circumferential angle θ [rad].
    dz : float
        Grid spacing in axial direction z [m].
    omega_rotor : float
        Rotor angular speed [rad/s],
        positive clockwise, negative counterclockwise.
    mu_a : float
        Dynamic viscosity μ_a [Pa·s].
    max_iter : int
        Maximum number of SOR iterations.
    tol : float
        Convergence tolerance on max update |p_new - p_old|.
    omega_relax : float
        SOR relaxation parameter (1.0 = Gauss–Seidel).
    bc_z : {"neumann", "dirichlet"}
        Type of boundary condition in z.

    Returns
    -------
    p : ndarray (Nz, Nθ)
        Pressure field p(θ, z) [Pa].
    """
    h = np.asarray(h)
    Nz, Nth = h.shape

    # Convert angular grid to arc length spacing in circumferential direction
    ds = R * dtheta           # [m]

    # NOTE: sign flip so that positive omega_rotor (clockwise) gives
    # negative surface speed in the +θ (counterclockwise) coordinate.
    u_l = -R * omega_rotor    # surface speed [m/s]

    p = np.zeros_like(h)
    c = h**3 / (12.0 * mu_a)

    for _ in range(max_iter):
        max_res = 0.0

        # Loop over axial interior nodes
        for j in range(1, Nz - 1):
            # Loop over ALL circumferential nodes with periodic neighbors
            for i in range(Nth):
                # Periodic indices in θ
                i_minus = (i - 1) % Nth
                i_plus = (i + 1) % Nth

                # Coefficients c at half steps
                c_imh = 0.5 * (c[j, i_minus] + c[j, i])
                c_iph = 0.5 * (c[j, i] + c[j, i_plus])
                c_jmh = 0.5 * (c[j - 1, i] + c[j, i])
                c_jph = 0.5 * (c[j, i] + c[j + 1, i])

                # ∂h/∂s using central difference in θ (via s = R θ)
                dhds = (h[j, i_plus] - h[j, i_minus]) / (2.0 * ds)
                rhs = (u_l / 2.0) * dhds

                # Neighbor pressures
                p_im = p[j, i_minus]
                p_ip = p[j, i_plus]
                p_jm = p[j - 1, i]
                p_jp = p[j + 1, i]

                # Discrete coefficients
                Ax = (c_iph + c_imh) / ds**2
                Ay = (c_jph + c_jmh) / dz**2
                A_center = Ax + Ay

                p_new = (
                    c_iph * p_ip / ds**2
                    + c_imh * p_im / ds**2
                    + c_jph * p_jp / dz**2
                    + c_jmh * p_jm / dz**2
                    - rhs
                ) / A_center

                p_old = p[j, i]
                p[j, i] = (1.0 - omega_relax) * p_old + omega_relax * p_new
                max_res = max(max_res, abs(p[j, i] - p_old))

        # z-boundaries
        if bc_z == "dirichlet":
            p[0, :] = 0.0
            p[-1, :] = 0.0
        else:  # "neumann"
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]

        if max_res < tol:
            break

    return p

import numpy as np
from utils import rpm_to_rads

from solve_K_C import solve_K_C
"""
D: diameter of bearing (m)
omega: angular velocity (rad/s)
eta: oil viscosity (Pa * s)
L: bearing length (m)
f: static load (N)
c: clearance between shaft and bearing (m)

"""
"""
Matrixes from page 200 example 5.5.1
"""
K_ex = np.array([
    [12.81, 16.39],
    [-25.06, 8.815]
]) * 1e6   # MN/m → N/m

C_ex = np.array([
    [232.9, -81.92],
    [-81.92, 294.9]
]) * 1e3   # kNs/m → Ns/m



D, omega, eta, L, f, c = 100 * 1e-3, rpm_to_rads(1500), 0.1, 30 * 1e-3, 525, 0.1 * 1e-3
K, C = solve_K_C(D, omega, eta, L, f, c)


print(K, "K")
print(C, "C")
"""
unit conversions back - should be accurate to at least 1 decimal
"""
dK = (K_ex - K) * 1e-6
dC = (C_ex - C) * 1e-3


"""
should be ≈ 0
"""
print("K_e - K =\n", dK)
print("\nC_e - C =\n", dC)

import math
import numpy as np

def compute_K(f, c, epsilon):
    """
    Calculates the stiffness matrix from the eccentricity, static load and clearance.
    
    Params:
    f: Static load (N)
    c: Clearance (mm)
    epsilon: Eccentricity

    returns: K_e matrix
    """
    h0 = _compute_h0(epsilon)

    auu = h0 * 4 *(math.pi**2 * (2 - epsilon**2) + 16 * epsilon**2)
    auv = h0 * (math.pi * (math.pi**2 * (1 - epsilon**2)**2) - 16 * epsilon**4) / (epsilon * math.sqrt(1 - epsilon**2))
    avu = -h0 * (math.pi * (math.pi**2 * (1 - epsilon**2) * (1 + 2 * epsilon**2) + 32 * epsilon**2) * (1 + epsilon**2)) / (epsilon * math.sqrt(1 - epsilon**2))
    avv = h0 * 4 * (math.pi**2 * (1 + 2 * epsilon**2) + (32 * epsilon**2 * (1 + epsilon**2)) / (1 - epsilon**2))
    arr = np.array([[auu, auv], [avu, avv]])
    return (f / c) * arr

def compute_C(f, c, omega, epsilon):
    """
    Calculates the damping matrix from the eccentricity, static load, shaft speed and clearance.
    
    Params:
    f: Static load (N)
    c: Clearance (mm)
    omega: Shaft speed (rad/s)
    epsilon: Eccentricity

    returns: C_e matrix
    """
    h0 = _compute_h0(epsilon)

    buu = h0 * (2 * math.pi * math.sqrt(1 - epsilon**2) * (math.pi**2 * (1 + 2 * epsilon**2) - 16 * epsilon**2)) / (epsilon)
    buv = -h0 * 8 * (math.pi**2 * (1 + 2 * epsilon**2) - 16 * epsilon**2)
    bvv = h0 * (2 * math.pi * (math.pi**2 * (1 - epsilon**2)**2) + 48 * epsilon**2) / (epsilon * math.sqrt(1 - epsilon**2))
    arr = np.array([[buu, buv], [buv, bvv]])
    return (f / (c * omega)) * arr

def _compute_h0(epsilon):
    return (1 / (math.pi**2 * (1 - epsilon**2) + 16 * epsilon**2)**(3 / 2))
from compute_s import compute_s
from compute_matrices import compute_K, compute_C
from eccentricity import solve_eccentricity
"""
Uses solvers for s, eccentricity, and compute K and compute C functions to provide end to end solution for K and C from initial values
"""
def solve_K_C(D, omega, eta, L, f, c):
    S = compute_s(D, omega, eta, L, f, c)
    eps = solve_eccentricity(S)
    K = compute_K(f, c, eps)
    C = compute_C(f, c, omega, eps)
    return K, C, eps
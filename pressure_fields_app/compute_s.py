def compute_s(D, omega, eta, L, f, c):
    """
    implementation from Friswell (5.84)
    """

    S = (D * omega * eta * (L**3)) / (8 * f * (c**2))
    return S
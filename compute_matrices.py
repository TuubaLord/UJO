import math

def compute_K(f, c, epsilon):
    h0 = _compute_h0(epsilon)

    auu = h0 * 4 *(math.pi**2 * (2 - epsilon**2) + 16 * epsilon**2)
    auv = h0 * (math.pi * (math.pi**2 * (1 - epsilon**2)**2) - 16 * epsilon**4) / (epsilon * math.sqrt(1 - epsilon**2))
    avu = -h0 * () / ()
    avv = 1
    return

def compute_C(f, c, omega, epsilon):
    pass

def _compute_h0(epsilon):
    return (1 / (math.pi**2 * (1 - epsilon**2) + 16 * epsilon**2)**(3 / 2))
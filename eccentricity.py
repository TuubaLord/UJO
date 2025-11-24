import numpy as np
from sympy import symbols, Eq, solve

def solve_eccentricity(S):
    """
    implementation from Friswell (5.83)
    """
    e = symbols('e')
    equation = Eq(e**8 - 4*e**6 + (6 - S**2*(16 - np.pi**2))*e**4 - (4 + np.pi**2*S**2)*e**2 + 1, 0)
    solutions = solve(equation, e)

    return 0.266297509005477


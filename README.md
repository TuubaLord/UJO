# UJO

Team UJO for Rotor x bearing hackaton

## Solving K and C matrixes, and eccentricity

run the file test_and_compute_K_C.py
-> this computes eccentricity, and solves the K and C matrixes as well as compares them to example 5.5.1


# Gepardointi


## Fluid bearing stiffness and damping matrices

The repository now includes `fluid_bearing.py`, which evaluates linearized
stiffness (K) and damping (C) matrices for fluid-film bearings using a
short-bearing analytical approximation.

### Governing equations (journal bearing)
For a journal bearing with radius $R$, axial length $L$, radial clearance $c$,
 lubricant dynamic viscosity $\mu$, and eccentricity ratio $\varepsilon = e/c$,
 the stiffness and damping base factors are

\begin{align}
 k_0 &= \frac{6 \pi \mu R^3 L}{c^3}, \\
 c_0 &= \frac{\pi \mu R^3 L}{2 c^2}.
\end{align}

The linearized short-bearing (Ocvirk) coefficients used in the code are

\begin{align}
 K_{xx} &= \frac{k_0\,\varepsilon(4+\varepsilon^2)}{4(1-\varepsilon^2)^2}, &
 K_{yy} &= \frac{k_0\,\varepsilon(4-\varepsilon^2)}{4(1-\varepsilon^2)^2},\\
 K_{xy} &= \frac{3 k_0 \varepsilon^2}{4(1-\varepsilon^2)^2}, &
 K_{yx} &= -K_{xy},\\[6pt]
 C_{xx} &= \frac{c_0(2-\varepsilon^2)}{(1-\varepsilon^2)^2}, &
 C_{yy} &= \frac{c_0(1+\varepsilon^2)}{(1-\varepsilon^2)^2},\\
 C_{xy} &= \frac{c_0\,\varepsilon}{(1-\varepsilon^2)^2}, &
 C_{yx} &= -C_{xy}.
\end{align}

### Tilting-pad simplification
Tilting-pad bearings are modeled as load-sharing short bearings with suppressed
cross-coupled terms. The total axial length is divided by the count of loaded
pads (``loaded_fraction * pad_count``), and only diagonal coefficients are
retained.

### Usage
Compute matrices directly from Python:

```python
from fluid_bearing import (
    BearingGeometry,
    journal_bearing_matrices,
    tilting_pad_matrices,
)

test_geo = BearingGeometry(
    radius=0.05,      # m
    length=0.04,      # m
    clearance=150e-6, # m
    viscosity=0.02,   # Pa·s
)

journal = journal_bearing_matrices(test_geo, eccentricity_ratio=0.6)
print("Journal K:\n", journal.stiffness)
print("Journal C:\n", journal.damping)

tilt = tilting_pad_matrices(test_geo, eccentricity_ratio=0.6, pad_count=5)
print("Tilting-pad K:\n", tilt.stiffness)
print("Tilting-pad C:\n", tilt.damping)
```

`journal_bearing_matrices` will raise `EccentricityOutOfRange` if
``eccentricity_ratio`` is not between 0 and 1. `tilting_pad_matrices` validates
pad counts and the ``loaded_fraction`` argument.

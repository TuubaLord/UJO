"""Utilities for computing linearized stiffness and damping matrices for fluid film bearings.

The current implementation uses short-bearing analytical approximations for
journal bearings and a simplified tilting-pad model that suppresses cross-coupled
terms.  The formulas are documented in the project README.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BearingGeometry:
    """Geometric and fluid properties of a bearing.

    Attributes
    ----------
    radius: Journal radius (m).
    length: Axial length of the bearing (m).
    clearance: Radial clearance (m).
    viscosity: Dynamic viscosity of the lubricant (Pa·s).
    """

    radius: float
    length: float
    clearance: float
    viscosity: float


@dataclass
class BearingMatrices:
    """Container for stiffness and damping matrices."""

    stiffness: List[List[float]]
    damping: List[List[float]]

    def as_tuple(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Return the matrices as a tuple for unpacking convenience."""

        return self.stiffness, self.damping


class EccentricityOutOfRange(ValueError):
    """Raised when the supplied eccentricity ratio is outside (0, 1)."""


_DEF_SMALL = 1e-12


def _validate_eccentricity_ratio(eccentricity_ratio: float) -> float:
    if not 0.0 < eccentricity_ratio < 1.0:
        raise EccentricityOutOfRange(
            "Eccentricity ratio must be between 0 and 1 (exclusive)."
        )
    return eccentricity_ratio


def journal_bearing_matrices(
    geometry: BearingGeometry, eccentricity_ratio: float
) -> BearingMatrices:
    """Compute stiffness and damping matrices for a journal bearing.

    The implementation follows analytical short-bearing approximations with
    linearized dynamic coefficients, using the expressions documented in the
    README.  Cross-coupled terms are retained to capture fluid-induced swirl.

    Parameters
    ----------
    geometry: BearingGeometry
        Physical dimensions and lubricant viscosity.
    eccentricity_ratio: float
        Ratio of journal center displacement to radial clearance (0 < e/c < 1).

    Returns
    -------
    BearingMatrices
        Stiffness (K) and damping (C) matrices in the XY-plane.
    """

    eps = _validate_eccentricity_ratio(eccentricity_ratio)
    denom = (1.0 - eps**2) ** 2 + _DEF_SMALL

    # Base scaling factors for stiffness and damping.
    k0 = (
        6.0
        * math.pi
        * geometry.viscosity
        * geometry.radius**3
        * geometry.length
        / (geometry.clearance**3)
    )
    c0 = (
        math.pi
        * geometry.viscosity
        * geometry.radius**3
        * geometry.length
        / (2.0 * geometry.clearance**2)
    )

    # Stiffness coefficients (short-bearing Ocvirk approximation).
    kxx = k0 * (eps * (4.0 + eps**2)) / (4.0 * denom)
    kyy = k0 * (eps * (4.0 - eps**2)) / (4.0 * denom)
    kxy = k0 * (3.0 * eps**2) / (4.0 * denom)
    kyx = -kxy

    # Damping coefficients (same short-bearing linearization).
    cxx = c0 * (2.0 - eps**2) / denom
    cyy = c0 * (1.0 + eps**2) / denom
    cxy = c0 * eps / denom
    cyx = -cxy

    stiffness = [[float(kxx), float(kxy)], [float(kyx), float(kyy)]]
    damping = [[float(cxx), float(cxy)], [float(cyx), float(cyy)]]

    return BearingMatrices(stiffness=stiffness, damping=damping)


def tilting_pad_matrices(
    geometry: BearingGeometry,
    eccentricity_ratio: float,
    pad_count: int = 4,
    loaded_fraction: float = 0.5,
) -> BearingMatrices:
    """Compute stiffness and damping matrices for an ideal tilting-pad bearing.

    The model assumes each loaded pad behaves like a short journal bearing but
    tilting action suppresses fluid cross-coupling.  The total load is shared
    across ``loaded_fraction * pad_count`` pads, reducing the effective axial
    length in the coefficient scaling.

    Parameters
    ----------
    geometry: BearingGeometry
        Physical dimensions and lubricant viscosity.
    eccentricity_ratio: float
        Ratio of journal center displacement to radial clearance (0 < e/c < 1).
    pad_count: int, optional
        Total number of pads. Defaults to 4.
    loaded_fraction: float, optional
        Fraction of pads that are load-carrying (typically 0.5). Must be within
        (0, 1].

    Returns
    -------
    BearingMatrices
        Stiffness (K) and damping (C) matrices in the XY-plane with zero
        cross-coupled terms.
    """

    if not 0.0 < loaded_fraction <= 1.0:
        raise ValueError("loaded_fraction must be in the range (0, 1].")
    if pad_count <= 0:
        raise ValueError("pad_count must be positive.")

    effective_pads = max(int(round(pad_count * loaded_fraction)), 1)
    pad_length = geometry.length / effective_pads
    pad_geometry = BearingGeometry(
        radius=geometry.radius,
        length=pad_length,
        clearance=geometry.clearance,
        viscosity=geometry.viscosity,
    )

    base = journal_bearing_matrices(pad_geometry, eccentricity_ratio)
    k_diag = [
        [base.stiffness[0][0], 0.0],
        [0.0, base.stiffness[1][1]],
    ]
    c_diag = [
        [base.damping[0][0], 0.0],
        [0.0, base.damping[1][1]],
    ]

    return BearingMatrices(stiffness=k_diag, damping=c_diag)

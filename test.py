from fluid_bearing import (
    BearingGeometry,
    journal_bearing_matrices,
    tilting_pad_matrices,
)


def main():
    geometry = BearingGeometry(
        radius=0.05,  # m
        length=0.04,  # m
        clearance=150e-6,  # m
        viscosity=0.02,  # Pa·s
    )

    journal = journal_bearing_matrices(geometry, eccentricity_ratio=0.6)
    print("Journal bearing stiffness matrix (N/m):\n", journal.stiffness)
    print("Journal bearing damping matrix (N·s/m):\n", journal.damping)

    tilt = tilting_pad_matrices(geometry, eccentricity_ratio=0.6, pad_count=5)
    print("\nTilting-pad stiffness matrix (N/m):\n", tilt.stiffness)
    print("Tilting-pad damping matrix (N·s/m):\n", tilt.damping)


if __name__ == "__main__":
    main()

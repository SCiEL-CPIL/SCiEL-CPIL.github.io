"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    s3 = math.sqrt(3)
    # 11 hexagons in a compact hexagonal cluster, all flat-top (0 deg rotation)
    # Axial coords (q, r) -> (1.5*q, s3*(r + 0.5*(q & 1)))
    axial = [
        (0, 0),
        (1, 0), (1, -1),
        (0, -1),
        (-1, 0), (-1, 1),
        (0, 1),
        (2, 0), (2, -1),
        (-2, 0), (-2, 1),
    ]
    inner = []
    for q, r in axial:
        x = 1.5 * q
        y = s3 * (r + 0.5 * (q % 2))
        inner.append([x, y, 0.0])
    # Outer hexagon: center (0,0), side length 4.0, rotation 0 deg
    return inner, (0.0, 0.0), 4.0, 0.0
# EVOLVE-BLOCK-END

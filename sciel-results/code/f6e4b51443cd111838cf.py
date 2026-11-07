"""Symbolic regression candidate: Complete quadratic polynomial basis.

This model uses a linear-in-parameters basis consisting of all monomials of
degree up to 2 in x, y, z. The basis functions are:
1, x, y, z, x^2, y^2, z^2, xy, xz, yz.

This representation is:
1. Convex in the parameters, ensuring BFGS finds the global minimum.
2. Complete for all quadratic functions.
3. Numerically stable for inputs in [0.5, 2].
4. A strong general-purpose baseline that captures linear trends,
   curvature, and pairwise interactions without the potential numerical
   pitfalls or redundancy of mixed rational-polynomial bases.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # 10-term basis: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    c1 = np.ones_like(x, dtype=np.float64)
    c2 = x
    c3 = y
    c4 = z
    c5 = x * x
    c6 = y * y
    c7 = z * z
    c8 = x * y
    c9 = x * z
    c10 = y * z

    out = (
        p[0] * c1
        + p[1] * c2
        + p[2] * c3
        + p[3] * c4
        + p[4] * c5
        + p[5] * c6
        + p[6] * c7
        + p[7] * c8
        + p[8] * c9
        + p[9] * c10
    )
    return out
# EVOLVE-BLOCK-END

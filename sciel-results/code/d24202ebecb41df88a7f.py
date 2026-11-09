# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    d = 11
    # Construct 16 points from an orthogonal cross-polytope in 11D.
    # Using e_i and -e_i for i=0..10 gives 22 points, all norm 1, all distinct,
    # and the minimum squared distance between any two points is 2 >= 1.
    pts = np.zeros((2 * d, d), dtype=np.int64)
    for i in range(d):
        pts[2 * i, i] = 1
        pts[2 * i + 1, i] = -1
    pts = np.around(pts).astype(np.int64)
    return pts


# EVOLVE-BLOCK-END

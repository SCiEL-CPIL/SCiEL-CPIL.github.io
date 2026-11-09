# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3) using the
    Hermitian-unital blocking construction. N is valid iff every point x of F_49^2
    lies on a line {x + t*v : t in F_49} with v = (1,m) or v = (0,1) whose points
    other than x all belong to N. We make N as small as possible.

    Returns:
        points: np.ndarray of shape (k, 2, 2) int64. Each point is [[xa, xb], [ya, yb]].
    """
    p = 7
    k = 3  # alpha^2 = 3 (smallest quadratic non-residue mod 7)

    # Y = {(u^2 mod p, v) : u, v in F_p}
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    # Excluded set S = points in U whose y is NOT in Y.
    # N = F_49^2 \ S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue  # excluded point
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END

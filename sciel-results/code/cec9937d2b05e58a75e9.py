# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points with squared norm 2
    (vectors having exactly two +/-1 entries). The minimum pairwise squared
    distance is 2, equal to the maximum squared norm, satisfying the constraint.
    """
    d = 11
    vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            # +1 at i, +1 at j
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = 1
            vectors.append(v)
            # +1 at i, -1 at j
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            vectors.append(v)
            # -1 at i, +1 at j
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = 1
            vectors.append(v)
            # -1 at i, -1 at j
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = -1
            vectors.append(v)

    points = np.array(vectors, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    return points


# EVOLVE-BLOCK-END

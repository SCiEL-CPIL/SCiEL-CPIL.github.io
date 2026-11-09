import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs the set of all integer vectors in Z^11 with squared norm 2.
    These are the vectors of the form (+/- 1, +/- 1, 0, ..., 0).
    
    The number of such vectors is 2 * C(11, 2) * 2 = 4 * 55 = 220.
    
    For any two distinct vectors p, q in this set:
    - If they have disjoint supports, ||p - q||^2 = 2 + 2 - 0 = 4.
    - If they have the same support, they differ by signs.
      - If one sign is flipped: p=(1,1,..), q=(1,-1,..). p-q=(0,2,..). ||p-q||^2 = 4.
      - If both signs are flipped: p=(1,1,..), q=(-1,-1,..). p-q=(2,2,..). ||p-q||^2 = 8.
      - Note: p and q are distinct, so they cannot be identical.
    - Wait, is it possible to have overlap of size 1?
      Yes. p=(1,1,0,..), q=(1,0,1,..).
      p and q have supports {0,1} and {0,2}. Intersection {0}.
      p - q = (0, 1, -1, ..).
      ||p - q||^2 = 0 + 1 + 1 = 2.
    
    So the minimum squared distance is 2.
    The maximum squared norm is 2.
    The condition min_dist^2 >= max_norm^2 becomes 2 >= 2, which is true.
    
    Thus, the set of all 220 weight-2 vectors is a valid configuration.
    """
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # Sign combinations for the two non-zero entries
            # (1, 1), (1, -1), (-1, 1), (-1, -1)
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    points.append(p)
    
    return np.array(points)

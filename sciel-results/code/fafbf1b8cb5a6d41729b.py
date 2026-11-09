# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    We define S (the complement of N) as the y-axis: x = 0.
    S = { [[0, 0], [y_a, y_b]] | y_a, y_b in {0, ..., 6} }
    |S| = 49.
    N = F_49^2 \ S.
    
    Validity Check:
    1. For s in S (x=0, y=y_s):
       The horizontal line y = y_s intersects S only at s (since other points have x != 0).
       So L \ {s} is subset of N.
    2. For n in N (x_n != 0, y_n):
       The vertical line x = x_n is disjoint from S (since S has x=0).
       So L is subset of N.
    """
    p = 7
    # Field size q = p^2 = 49
    # Points are represented as [[xa, xb], [ya, yb]]
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # S is the y-axis: xa == 0 and xb == 0
                    if xa == 0 and xb == 0:
                        continue # This point is in S, so NOT in N
                    points.append([[xa, xb], [ya, yb]])
                    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import time

import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    We choose the complement C to be a line of q=49 points.
    C = { (x, 0) for x in F_49 }.
    This ensures:
    1. For any x in C, there exists a line through x (e.g., vertical) containing no other points of C.
    2. For any x in N, since |C| = 49 < 50 (number of lines through x), there exists a line through x disjoint from C.
    
    Thus N = F_49^2 \ C is a valid Nikodym set.
    |N| = 2401 - 49 = 2352.
    Score = 49.
    """
    p = 7
    q = p * p
    
    # Identify points in C: y = 0 => y_a = 0, y_b = 0
    # C = { [[x_a, x_b], [0, 0]] for x_a in 0..6, x_b in 0..6 }
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # If y is (0,0), it is in C, so skip (do not add to N)
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END

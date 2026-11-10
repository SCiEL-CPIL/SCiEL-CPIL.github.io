# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49.
    We choose the excluded set S to be the Hermitian unital U.
    U = { (x, y) : Trace(x) + Norm(y) = 0 }.
    Trace(x) = 2 * x_a.
    Norm(y) = y_a^2 - k * y_b^2, where k is a quadratic non-residue mod 7.
    For p=7, k=3.
    The Nikodym set is N = F_q^2 \ S.
    This N is valid because:
    1. For x in U, the tangent line at x intersects U only at x, so the rest of the line is in N.
    2. For x not in U, there exists a line through x disjoint from U (external line), so the rest of the line is in N.
    The size of S is q^(3/2) + 1 = 344, which is the maximum possible size for such a set.
    """
    p = 7
    q = p * p
    
    # Smallest quadratic non-residue mod 7
    k = 3
    
    points = []
    for xa in range(p):
        for xb in range(p):
            # x = xa + xb * alpha
            # Trace(x) = x + x^p = 2 * xa
            tr_x = (2 * xa) % p
            
            for ya in range(p):
                for yb in range(p):
                    # y = ya + yb * alpha
                    # Norm(y) = y * y^p = (ya + yb*alpha)(ya + yb*alpha^p)
                    # alpha^p = -alpha (since alpha^2=3, alpha^7 = alpha*(alpha^2)^3 = alpha*27 = alpha*6 = -alpha)
                    # Norm(y) = (ya + yb*alpha)(ya - yb*alpha) = ya^2 - yb^2 * alpha^2 = ya^2 - 3 * yb^2
                    norm_y = (ya * ya - k * yb * yb) % p
                    
                    # Check if point is in U
                    if (tr_x + norm_y) % p == 0:
                        continue # Excluded point
                    
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49 (p=7).
    We use the complement of the set U defined by the equation:
    2 * xa + ya^2 - yb^2 * k = 0 (mod p)
    where k=3 is the smallest quadratic non-residue.
    
    The set U has size q^2 = 49.
    We hypothesize that U is a valid complement for a Nikodym set,
    meaning for every x in U, there exists a direction v such that
    the line through x in direction v intersects U only at x.
    
    The Nikodym set N is F_q^2 \ U.
    Size of N = 2401 - 49 = 2352.
    Complement size = 49.
    """
    p = 7
    q = p * p
    
    # Smallest quadratic non-residue mod 7
    # Squares mod 7: 0, 1, 4, 2 (1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1)
    # Non-residues: 3, 5, 6
    k = 3
    
    points = []
    
    # Iterate through all points in F_q^2
    # x = (xa, xb), y = (ya, yb)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if point is in U
                    # Equation: 2 * xa + ya^2 - yb^2 * k == 0 (mod p)
                    val = (2 * xa + ya * ya - yb * yb * k) % p
                    if val != 0:
                        # Point is NOT in U, so it IS in N
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END

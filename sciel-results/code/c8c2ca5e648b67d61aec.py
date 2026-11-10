import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Find k such that alpha^2 = k is a non-residue
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
            
    # Precompute field operations or just use the pair representation
    # Point (xa, xb, ya, yb)
    # Condition: 2*xa + ya*ya - yb*yb*k == 0 (mod p)
    
    excluded = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        excluded.append([[xa, xb], [ya, yb]])
                        
    # The Nikodym set N is the complement of S (excluded)
    # But the function returns N.
    # Wait, the task says "returning ... listing the points of a set N".
    # "N must be a Nikodym set".
    # "Metric: complement size 2401 - |N|".
    # So we want N to be small.
    # S is the complement.
    # So we return the points NOT in excluded.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p != 0:
                        points.append([[xa, xb], [ya, yb]])
                        
    return np.array(points, dtype=np.int64)

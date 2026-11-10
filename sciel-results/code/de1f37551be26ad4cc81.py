# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if in Unital
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    if not in_unital:
                        points.append([[xa, xb], [ya, yb]])
                        
    return np.array(points, dtype=np.int64)
# EVOLVE-BLOCK-END

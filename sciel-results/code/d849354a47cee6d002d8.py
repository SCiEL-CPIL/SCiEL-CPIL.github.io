# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # The complement C is a single point: (0, 0) in F_49^2
    # N = F_49^2 \ C
    # This is a valid Nikodym set with |N| = 2400
    # For any x != (0,0), there exist lines through x not containing (0,0),
    # so those lines have all other points in N.
    # For x = (0,0), any line through it has all other points in N.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Skip the single complement point (0,0)
                    if xa == 0 and xb == 0 and ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END

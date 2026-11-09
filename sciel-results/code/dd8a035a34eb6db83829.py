import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3 # alpha^2 = 3
    
    # Precompute squares for all elements in F_49
    # Element is (a, b)
    squares = {}
    for a in range(p):
        for b in range(p):
            # (a + b*alpha)^2 = a^2 + 2ab*alpha + b^2*alpha^2
            # = a^2 + 2ab*alpha + 3b^2
            # = (a^2 + 3b^2) + (2ab)*alpha
            ya = (a*a + 3*b*b) % p
            yb = (2*a*b) % p
            squares[(a, b)] = (ya, yb)
            
    # Construct the set S (excluded points)
    # S = { (x, x^2) | x in F_49 }
    S = set()
    for a in range(p):
        for b in range(p):
            ya, yb = squares[(a, b)]
            S.add(((a, b), (ya, yb)))
            
    # Construct N (included points)
    # N = F_49^2 \ S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if ((xa, xb), (ya, yb)) not in S:
                        points.append([[xa, xb], [ya, yb]])
                        
    return np.array(points, dtype=np.int64)

import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    We choose S to be the line y=0 in F_49^2.
    S = { (x, 0) : x in F_49 }.
    N = F_49^2 \ S.
    This N is a valid Nikodym set.
    """
    p = 7
    q = p * p
    
    # Identify the line y=0.
    # A point is (x, y) where x, y are in F_q.
    # y=0 means y_a=0 and y_b=0.
    # x can be any (x_a, x_b).
    
    # We return N.
    # N contains all points except those with y_a=0, y_b=0.
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if point is in S (y=0)
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

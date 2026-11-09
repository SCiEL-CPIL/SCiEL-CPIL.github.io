# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where the complement S is the parabola
    {(x, x^2) : x in F_49}. For each point s in S, the tangent line at s
    intersects S only at s, satisfying the Nikodym condition.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3. Element (a, b) = a + b*alpha.
    Returns N = F_49^2 \\ S as integer array of shape (k, 2, 2).
    """
    p = 7
    k = 3  # alpha^2 = 3 in F_7
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = k
    def add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def mul(x, y):
        a, b = x
        c, d = y
        # (a + b*alpha)(c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # = (ac + bd*k) + (ad + bc)*alpha
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)
    
    def neg(x):
        return ((-x[0]) % p, (-x[1]) % p)
    
    # Enumerate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Build the parabola S = {(x, x^2) : x in F_49}
    S = set()
    for x in field_elements:
        x2 = mul(x, x)
        S.add((x, x2))
    
    # N = all points not in S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = (xa, xb)
                    y = (ya, yb)
                    if (x, y) not in S:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END

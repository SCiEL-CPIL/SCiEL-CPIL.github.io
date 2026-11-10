# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using the complement of a parabola.
    
    Let S be the set of points on the parabola y = x^2 in F_49^2.
    S has size 49. Any line intersects S in at most 2 points.
    For x in S, the tangent line at x intersects S only at x.
    For x not in S, there exists a line through x disjoint from S.
    Thus N = F_49^2 \ S is a Nikodym set.
    The complement size is 49, which is the maximum possible for an arc in AG(2, 49).
    """
    p = 7
    q = p * p
    # alpha^2 = 3
    k = 3
    
    # Precompute field operations for speed, though for q=49 it's fast enough
    # We represent elements as (a, b) corresponding to a + b*alpha
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + a0*b1*alpha + a1*alpha*b0 + a1*b1*alpha^2
        # alpha^2 = 3
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_pow(a, exp):
        res = (1, 0)
        base = a
        while exp:
            if exp % 2 == 1:
                res = field_mul(res, base)
            base = field_mul(base, base)
            exp //= 2
        return res
    
    # Generate all points in F_49
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Generate the parabola S = {(t, t^2) : t in F_49}
    S = set()
    for t in field_elements:
        t_sq = field_pow(t, 2)
        # Point is (x, y) where x=t, y=t^2
        # Representation: x = (x_a, x_b), y = (y_a, y_b)
        S.add((t[0], t[1], t_sq[0], t_sq[1]))
        
    # Generate N = F_49^2 \ S
    # N is a list of points not in S
    N_points = []
    for x in field_elements:
        for y in field_elements:
            pt = (x[0], x[1], y[0], y[1])
            if pt not in S:
                N_points.append(pt)
                
    # Convert to numpy array
    # Shape (k, 2, 2)
    # Each point is [[x_a, x_b], [y_a, y_b]]
    # The task says: "A point of F_49^2 is written [[x_a, x_b], [y_a, y_b]]"
    # So row 0 is x coords, row 1 is y coords.
    
    if not N_points:
        # Fallback, should not happen
        return np.empty((0, 2, 2), dtype=np.int64)
        
    arr = np.array(N_points, dtype=np.int64).reshape(-1, 2, 2)
    
    return arr


# EVOLVE-BLOCK-END

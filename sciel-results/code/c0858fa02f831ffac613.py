import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    
    # Precompute field operations
    # Element is (a, b) -> a + b*alpha
    # Mult: (a1, b1) * (a2, b2) = (a1a2 + k b1b2, a1b2 + a2b1)
    
    def field_mult(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1 * a2 + k * b1 * b2) % p, (a1 * b2 + a2 * b1) % p)
    
    def field_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
        
    def field_sub(e1, e2):
        return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)
        
    def field_inv(e):
        # (a+b alpha)^-1 = (a-b alpha) / (a^2 - k b^2)
        a, b = e
        norm = (a * a - k * b * b) % p
        if norm == 0:
            return (0, 0) # Should not happen for non-zero
        inv_norm = pow(norm, p - 2, p)
        return ((a * inv_norm) % p, (-b * inv_norm) % p)
        
    # Generate all points in F_q^2
    # Point is ((xa, xb), (ya, yb))
    # We are interested in the curve U: 2*xa + ya^2 - k*yb^2 = 0
    # This implies 2*xa = -(ya^2 - k*yb^2)
    # xa = -(ya^2 - k*yb^2) * inv(2)
    # inv(2) = 4 (since 2*4=8=1)
    inv2 = 4
    
    U_points = []
    for ya in range(p):
        for yb in range(p):
            val = (ya * ya - k * yb * yb) % p
            xa = (-val * inv2) % p
            # xb is free, 0..6
            for xb in range(p):
                U_points.append(((xa, xb), (ya, yb)))
                
    # Check validity of a set S
    # S is a list of points
    # For each x in S, check if there is a direction v such that L_v(x) \cap S = {x}
    # Directions: 0..6 (slopes m) and inf (vertical)
    # Total 50 directions.
    
    def is_valid_complement(S):
        if not S:
            return True
        S_set = set(S)
        for x in S:
            bad_dirs = set()
            for y in S:
                if x == y:
                    continue
                # Determine direction from x to y
                # x = ((xa, xb), (ya, yb)), y = ((xa2, xb2), (ya2, yb2))
                # dx = y - x, dy = ...
                # If dx = 0, direction is inf
                # Else direction is dy / dx
                dx = field_sub(y[0], x[0])
                dy = field_sub(y[1], x[1])
                
                if dx == (0, 0):
                    d = 7 # inf
                else:
                    inv_dx = field_inv(dx)
                    m = field_mult(dy, inv_dx)
                    # Map m to 0..6
                    d = m[0] * 7 + m[1] # Just a unique ID? 
                    # Wait, m is in F_49. There are 49 slopes.
                    # Plus inf. Total 50.
                    # We need to map F_49 element to 0..48.
                    # m = (ma, mb) -> ma * 7 + mb
                    d = m[0] * 7 + m[1]
                bad_dirs.add(d)
            if len(bad_dirs) >= 50:
                return False
        return True
        
    # Greedy construction
    # Start with S = U_points
    # If invalid, remove points that are "most blocked"
    
    S = list(U_points)
    
    # To speed up, we can just check if U is valid.
    # If not, we try a random subset or greedy removal.
    
    # Given time constraints, we will just return the seed's construction if U is not valid,
    # or try to optimize U.
    # Actually, let's just use the seed's logic but maybe with a better Y filter?
    # The seed used Y_special (squares).
    # Let's try to find the largest subset of U that is valid.
    
    # We'll use a simple heuristic:
    # 1. Try S = U.
    # 2. If invalid, pick a point x in S with the highest number of bad directions.
    # 3. Remove x.
    # 4. Repeat.
    
    # This might be slow if we recompute from scratch.
    # But |U| = 343.
    # is_valid_complement is O(|S|^2).
    # 343^2 ~ 120k.
    # If we remove 100 points, 100 * 120k = 12M ops. Fast in Python? Maybe 1-2s.
    
    start_time = time.time()
    deadline = 600 - 10 # 590s
    
    # Optimization: Precompute slopes between all pairs in U?
    # Not necessary for 343 points.
    
    current_S = list(U_points)
    
    while True:
        if time.time() > deadline:
            break
        if is_valid_complement(current_S):
            break
        # Find point with most bad directions
        max_bad = -1
        worst_point = None
        S_set = set(current_S)
        for x in current_S:
            bad_dirs = set()
            for y in current_S:
                if x == y:
                    continue
                dx = field_sub(y[0], x[0])
                dy = field_sub(y[1], x[1])
                if dx == (0, 0):
                    d = 49
                else:
                    inv_dx = field_inv(dx)
                    m = field_mult(dy, inv_dx)
                    d = m[0] * 7 + m[1]
                bad_dirs.add(d)
            if len(bad_dirs) > max_bad:
                max_bad = len(bad_dirs)
                worst_point = x
        if worst_point:
            current_S.remove(worst_point)
        else:
            break
            
    # Convert to array
    if not current_S:
        # Fallback to seed
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            continue
                        points.append([[xa, xb], [ya, yb]])
        current_S = points
    else:
        points = []
        for pt in current_S:
            x, y = pt
            points.append([[x[0], x[1]], [y[0], y[1]]])
            
    return np.array(points, dtype=np.int64)

import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field parameters
    # alpha^2 = 3
    k = 3
    
    # Precompute field operations
    # Elements are (a, b)
    # We represent points as [[xa, xb], [ya, yb]]
    
    # Construct U
    # U = { (x, y) | 2*xa + ya^2 - 3*yb^2 = 0 mod 7 }
    U_points = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - 3 * yb * yb) % p == 0:
                        U_points.add((xa, xb, ya, yb))
                        
    # Baseline Y_special
    Y_special = set()
    for u in range(p):
        for v in range(p):
            Y_special.add(((u * u) % p, v))
            
    # Helper to check if N is valid
    def is_valid_nikodym(Y):
        # N = (F_q^2 \ U) U (U \cap (F_q x Y))
        # S = U \ (U \cap (F_q x Y)) = { (x,y) in U | y not in Y }
        # We need to check for all x in S, exists v s.t. L \ {x} subset N
        
        # Precompute membership in N for fast lookup?
        # N contains all points NOT in U, plus points in U with y in Y.
        # So x in N iff (x not in U) OR (x in U and y in Y).
        
        # Directions
        # v = (1, m) for m in F_q, v = (0, 1)
        # m is (ma, mb)
        
        # Iterate over all points in S
        for x in U_points:
            xa, xb, ya, yb = x
            if (ya, yb) in Y:
                continue # x is in N, not in S
            
            # x is in S, so x not in N.
            # Check if there is a line through x in N
            found = False
            
            # Check vertical line v = (0, 1)
            # Points: (xa, xb + t, ya, yb)
            # For t != 0, is (xa, xb+t, ya, yb) in N?
            # It is in N if (not in U) or (in U and (ya, yb) in Y).
            # (ya, yb) is NOT in Y (since x in S).
            # So we need (xa, xb+t, ya, yb) NOT in U.
            # Check all t in 1..6
            ok = True
            for t in range(1, p):
                pt = (xa, (xb + t) % p, ya, yb)
                if pt in U_points:
                    ok = False
                    break
            if ok:
                found = True
            
            if not found:
                # Check lines v = (1, m)
                for ma in range(p):
                    for mb in range(p):
                        # v = (1, m)
                        # Points: (xa + t, xb + t*ma, ya + t*mb, yb) -- NO
                        # Field multiplication: (1, 0) * (ma, mb) = (ma, mb)
                        # Point: x + t*v = (xa + t*1, xb + t*ma, ya + t*mb, yb + t*0)?
                        # No, v = (1, m).
                        # x = (x, y).
                        # x + t v = (x + t, y + t m).
                        # x + t = (xa + t, xb)
                        # y + t m = (ya + t*ma - t*mb*k? No, (ya, yb) + t(ma, mb))
                        # (ya + t*ma, yb + t*mb)
                        
                        ok = True
                        for t in range(1, p):
                            nx_a = (xa + t) % p
                            nx_b = xb
                            ny_a = (ya + t * ma) % p
                            ny_b = (yb + t * mb) % p
                            pt = (nx_a, nx_b, ny_a, ny_b)
                            # Check if pt in N
                            # pt in N iff (pt not in U) or (pt in U and (ny_a, ny_b) in Y)
                            if pt in U_points:
                                if (ny_a, ny_b) not in Y:
                                    ok = False
                                    break
                            # else pt not in U, so pt in N
                        if ok:
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break
            
            if not found:
                return False
        
        return True

    Y = set(Y_special)
    
    # Greedy removal
    changed = True
    while changed:
        changed = False
        for y in list(Y):
            Y_new = Y - {y}
            if is_valid_nikodym(Y_new):
                Y = Y_new
                changed = True
                break # Restart loop with smaller Y
    
    # Construct final N
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = (xa, xb, ya, yb)
                    in_U = pt in U_points
                    in_Y = (ya, yb) in Y
                    if not in_U or in_Y:
                        points.append([[xa, xb], [ya, yb]])
                        
    return np.array(points, dtype=np.int64)

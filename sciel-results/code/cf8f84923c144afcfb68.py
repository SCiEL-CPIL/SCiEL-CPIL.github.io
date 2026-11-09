# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # alpha^2 = 3 is a non-residue mod 7 (3^3 = 27 = 6 = -1)
    k = 3
    
    # Precompute field operations
    # Element is (a, b) representing a + b*alpha
    # Multiplication: (a, b) * (c, d) = (ac + bd*k, ad + bc)
    
    # Unital U: Tr(x) + Norm(y) = 0
    # Tr(x) = x_a + x_a^2? No.
    # Trace from F_q to F_p: Tr(a + b*alpha) = a + a*alpha^2? No.
    # Tr(z) = z + z^p.
    # (a + b*alpha)^7 = a^7 + b^7*alpha^7 = a + b*alpha^7.
    # alpha^7 = alpha^(2*3 + 1) = alpha * (alpha^2)^3 = alpha * 3^3 = alpha * 27 = 6*alpha = -alpha.
    # So Tr(a + b*alpha) = a + b*alpha + a - b*alpha = 2a.
    # Norm(z) = z * z^p = (a + b*alpha)(a - b*alpha) = a^2 - b^2*alpha^2 = a^2 - 3b^2.
    
    # U = { (x, y) : 2*x_a + y_a^2 - 3*y_b^2 = 0 }
    
    # List all points in U
    U_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - 3 * yb * yb) % p == 0:
                        U_points.append(((xa, xb), (ya, yb)))
    
    # Total points in F_49^2
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    # Map point to index for fast lookup
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    u_set = set(U_points)
    
    # Initialize B (points in N) as empty
    B = set()
    C = set(U_points) # Complement N = F^2 \ C? No.
    # N is the Nikodym set. C is the complement.
    # We want to maximize |C|.
    # C is a subset of U.
    # Points not in U are always in N.
    
    # Condition 2: For all x not in C, there exists a line through x disjoint from C.
    # x not in C means x in B or x not in U.
    
    # Helper: Check if a line through x in direction v is disjoint from C
    # Line: x + t*v for t in F_49.
    # v is (1, m) or (0, 1).
    
    def line_disjoint_from_C(x, v, C_set):
        x_pt = x
        for t in range(q):
            if t == 0: continue # x itself, but x not in C
            # x + t*v
            # v = (v_x, v_y)
            # t*v = (t*v_x, t*v_y)
            # x + t*v = (x_x + t*v_x, x_y + t*v_y)
            # Field addition
            tx_a = (x_pt[0][0] + t * v[0][0]) % p
            tx_b = (x_pt[0][1] + t * v[0][1]) % p
            ty_a = (x_pt[1][0] + t * v[1][0]) % p
            ty_b = (x_pt[1][1] + t * v[1][1]) % p
            pt = ((tx_a, tx_b), (ty_a, ty_b))
            if pt in C_set:
                return False
        return True

    # Precompute directions
    # v = (1, m) for m in F_49, and v = (0, 1)
    # m = (m_a, m_b)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb))) # (1, m)
    directions.append(((0, 0), (0, 1))) # (0, 1)
    
    # Greedy loop
    # Start with C = U, B = empty
    C_set = set(U_points)
    B_set = set()
    
    # Uncovered points: x not in C such that no line through x is disjoint from C
    # We need to cover all x in (F^2 \ C)
    
    # Optimization: 
    # If we remove y from C (add to B), lines through y that intersected C only at y become disjoint.
    # These lines cover all points on them.
    
    # We iterate until all x not in C are covered.
    
    max_iter = 100
    for _ in range(max_iter):
        # Find uncovered points
        uncovered = []
        # Check all x not in C
        # This is expensive. 
        # x not in C means x in B or x not in U.
        
        # Let's check x not in U first, as there are many.
        # And x in B.
        
        # Actually, we just need to find ONE uncovered point to fix.
        found_uncovered = False
        for x in all_points:
            if x in C_set:
                continue
            # x is not in C. Check if covered.
            covered = False
            for v in directions:
                if line_disjoint_from_C(x, v, C_set):
                    covered = True
                    break
            if not covered:
                found_uncovered = True
                uncovered_x = x
                break
        
        if not found_uncovered:
            break
            
        # Fix uncovered_x
        # Find a line through uncovered_x that intersects C in exactly 1 point
        # And that point y is in C.
        # If we remove y, the line becomes disjoint.
        
        best_y = None
        for v in directions:
            # Intersections of line through uncovered_x in direction v with C
            inter = []
            for t in range(q):
                if t == 0: continue
                tx_a = (uncovered_x[0][0] + t * v[0][0]) % p
                tx_b = (uncovered_x[0][1] + t * v[0][1]) % p
                ty_a = (uncovered_x[1][0] + t * v[1][0]) % p
                ty_b = (uncovered_x[1][1] + t * v[1][1]) % p
                pt = ((tx_a, tx_b), (ty_a, ty_b))
                if pt in C_set:
                    inter.append(pt)
            if len(inter) == 1:
                best_y = inter[0]
                break
                
        if best_y is not None:
            C_set.remove(best_y)
            B_set.add(best_y)
        else:
            # No line intersects C in exactly 1 point?
            # This might mean we are stuck or need to remove more.
            # For safety, break or remove arbitrary.
            break
            
    # Construct N = F^2 \ C_set
    N_points = []
    for pt in all_points:
        if pt not in C_set:
            N_points.append([pt[0], pt[1]])
            
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END

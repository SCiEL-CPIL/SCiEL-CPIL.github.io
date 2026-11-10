import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy local search 
    starting from a union of 3 parabolas.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    p = 7
    q = p * p # 49
    k = 3 # alpha^2 = 3
    
    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul(a, b):
        return ((a[0]*b[0] + k*a[1]*b[1]) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def to_int(a):
        return a[0] + a[1]*p

    # Generate all points
    points = []
    for a in range(p):
        for b in range(p):
            points.append((a, b))
            
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    idx_to_point = {i: pt for i, pt in enumerate(points)}
    
    # Directions: (1, m) for m in F_q, and (0, 1)
    directions = [(1, m) for m in points] + [(0, 1)]
    # Note: (1, m) where m is a field element.
    # We need to represent m as (ma, mb).
    # So direction is a vector (dx, dy) where dx, dy are field elements.
    # v = (1, m) -> ( (1,0), (ma, mb) )
    # v = (0, 1) -> ( (0,0), (1,0) )
    
    dir_vectors = []
    for m in points:
        dir_vectors.append( ((1,0), m) )
    dir_vectors.append( ((0,0), (1,0)) )
    
    # Precompute lines
    # lines[idx][d_idx] = list of point indices on the line
    # To save memory, we can just compute on the fly or store sets.
    # N=2401, D=50. 2401*50*49 ints is too much.
    # Store as sets of indices.
    
    lines = [None] * q
    for i in range(q):
        P = idx_to_point[i]
        lines[i] = [None] * len(dir_vectors)
        for d_idx, v in enumerate(dir_vectors):
            line_pts = []
            for t in points:
                # Q = P + t*v
                # v = (vx, vy)
                vx, vy = v
                # t*v = (t*vx, t*vy)
                tv = (mul(t, vx), mul(t, vy))
                Q = add(P, tv)
                line_pts.append(point_to_idx[Q])
            lines[i][d_idx] = line_pts
            
    # Initialize C with 3 parabolas
    # y = x^2 + c for c in {0, 1, 2}
    # Check if c={0,1,2} works.
    # Diffs: 1, 2, 1. All squares.
    C_indices = set()
    for c_val in [0, 1, 2]:
        c = (c_val, 0)
        for x in points:
            y = add(mul(x, x), c)
            pt = (x, y)
            C_indices.add(point_to_idx[pt])
            
    # valid_counts[i] = number of valid directions for point i
    valid_counts = [0] * q
    
    def is_valid_dir(i, d_idx, C_set):
        P = idx_to_point[i]
        line_pts = lines[i][d_idx]
        # Check if line is compatible
        # If P in C: line \ {P} disjoint from C
        # If P not in C: line disjoint from C
        in_C = i in C_set
        for j in line_pts:
            if j == i:
                continue
            if j in C_set:
                if in_C:
                    return False
                else:
                    return False # Wait, if P not in C, line must be empty of C.
                # Actually:
                # If P in C, we allow P. Others must not be in C.
                # If P not in C, NO one can be in C.
                
    # Recalculate logic for is_valid_dir
    def count_valid_dirs(i, C_set):
        count = 0
        P_in_C = i in C_set
        for d_idx in range(len(dir_vectors)):
            line_pts = lines[i][d_idx]
            ok = True
            for j in line_pts:
                if j == i:
                    continue
                if j in C_set:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    # Initialize valid_counts
    for i in range(q):
        valid_counts[i] = count_valid_dirs(i, C_indices)
        
    # Check if initial set is valid
    if any(c == 0 for c in valid_counts):
        # Fallback to empty set or single parabola if 3 parabolas fail
        # But 3 parabolas should work.
        pass

    best_C = set(C_indices)
    
    # Greedy search
    # Candidates: points not in C with valid_counts > 0
    while time.time() < deadline - safety_margin:
        candidates = []
        for i in range(q):
            if i not in C_indices and valid_counts[i] > 0:
                candidates.append(i)
                
        if not candidates:
            break
            
        # Evaluate each candidate
        best_candidate = -1
        min_breaks = q + 1
        
        for P in candidates:
            # Simulate adding P
            new_C = C_indices | {P}
            breaks = 0
            # For each Q, check if any valid direction is broken
            # A direction d for Q is broken if P is on line L_Q^d and that line was previously empty of C (or only Q)
            # Actually, we just need to check if valid_counts[Q] becomes 0.
            # We can just recount for affected Qs.
            # Affected Qs are those where P is on a line with Q.
            # P is on L_Q^d iff Q is on L_P^d (lines are symmetric).
            # So for each d, for each Q in lines[P][d], check Q.
            
            affected_Qs = set()
            for d_idx in range(len(dir_vectors)):
                for Q in lines[P][d_idx]:
                    if Q != P:
                        affected_Qs.add(Q)
                        
            # Also P itself changes status
            affected_Qs.add(P)
            
            for Q in affected_Qs:
                old_count = valid_counts[Q]
                new_count = count_valid_dirs(Q, new_C)
                if new_count == 0 and old_count > 0:
                    breaks += 1
                    
            if breaks < min_breaks:
                min_breaks = breaks
                best_candidate = P
                
        if best_candidate == -1 or min_breaks > 0:
            # If we can't add any point without breaking others, stop
            # Or try a point that breaks 0?
            if min_breaks > 0:
                break
            # If min_breaks == 0, we can add best_candidate
            
        # Add best_candidate
        C_indices.add(best_candidate)
        
        # Update valid_counts for affected points
        P = best_candidate
        affected_Qs = set()
        for d_idx in range(len(dir_vectors)):
            for Q in lines[P][d_idx]:
                if Q != P:
                    affected_Qs.add(Q)
        affected_Qs.add(P)
        
        for Q in affected_Qs:
            valid_counts[Q] = count_valid_dirs(Q, C_indices)
            
        best_C = set(C_indices)
        
    # Construct output array
    N_points = []
    for i in range(q):
        if i not in best_C:
            pt = idx_to_point[i]
            N_points.append([[pt[0], pt[1]], [pt[2], pt[3]]] if len(pt)==4 else [[pt[0], pt[1]], [pt[0], pt[1]]])
            # pt is (x, y) where x=(xa,xb), y=(ya,yb)
            # pt = ((xa, xb), (ya, yb))
            x, y = pt
            N_points.append([[x[0], x[1]], [y[0], y[1]]])
            
    return np.array(N_points, dtype=np.int64)


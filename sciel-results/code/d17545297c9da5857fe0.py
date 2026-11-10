import numpy as np
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field operations
    # alpha^2 = 3
    k = 3
    
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def sub_f(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0b0 + a0b1*alpha + a1b0*alpha + a1b1*alpha^2
        # = (a0b0 + a1b1*k) + (a0b1 + a1b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def div_f(a, b):
        # a / b = a * b^{-1}
        # b = b0 + b1*alpha
        # b * (b0 - b1*alpha) = b0^2 - b1^2*k
        inv_denom = (b[0]*b[0] - b[1]*b[1]*k) % p
        if inv_denom == 0:
            return None # Should not happen for b != 0
        inv_denom_inv = pow(inv_denom, p-2, p)
        # b^{-1} = (b0 - b1*alpha) / (b0^2 - b1^2*k)
        inv_b = ((b[0] * inv_denom_inv) % p, (-b[1] * inv_denom_inv) % p)
        return mul_f(a, inv_b)

    # Precompute all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    n_points = len(points)
    
    # Precompute lines
    # Directions: m in F_q and infinity
    # Represent direction as (1, m) or (0, 1)
    # m is a pair (ma, mb)
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append(((1, 0), (ma, mb))) # (1, m)
    dirs.append(((0, 1), (0, 0))) # (0, 1)
    
    # For each point y, and each direction d, find the set of points on the line
    # Actually, we just need to know which points are on the line.
    # line[y][d] = list of points on line through y with direction d
    
    # To save memory/time, we can compute on the fly or precompute indices.
    # Let's precompute the list of point indices for each (y, d).
    
    # Map point to index
    point_to_idx = {}
    for i, pt in enumerate(points):
        point_to_idx[pt] = i
        
    # lines_through[y_idx][d_idx] = list of point indices
    lines_through = [[[] for _ in range(len(dirs))] for _ in range(n_points)]
    
    for y_idx, y in enumerate(points):
        for d_idx, dir_vec in enumerate(dirs):
            # Line: y + t * dir_vec
            # dir_vec = (dx, dy) where dx, dy are in F_q
            dx, dy = dir_vec
            line_pts = []
            for t_a in range(p):
                for t_b in range(p):
                    t = (t_a, t_b)
                    # pt = y + t * dir_vec
                    # t * dir_vec = (t*dx, t*dy)
                    tdx = mul_f(t, dx)
                    tdy = mul_f(t, dy)
                    pt = add_f(y[0], tdx)
                    pt_y = add_f(y[1], tdy)
                    pt_full = (pt, pt_y)
                    if pt_full in point_to_idx:
                        line_pts.append(point_to_idx[pt_full])
            lines_through[y_idx][d_idx] = line_pts
            
    # Greedy algorithm
    best_S = set()
    best_size = 0
    
    for restart in range(10): # 10 restarts
        S = set()
        # counts[y][d] = number of points in S on line through y with direction d
        counts = np.zeros((n_points, len(dirs)), dtype=np.int8)
        
        # valid[y] = True if y is currently valid
        valid = np.ones(n_points, dtype=bool)
        
        # Order of points to try
        order = list(range(n_points))
        random.shuffle(order)
        
        for x_idx in order:
            # Check if adding x_idx to S keeps all points valid
            # We need to check all y that are on lines through x_idx
            # These are the y's whose counts will change
            
            affected_y = set()
            for d_idx in range(len(dirs)):
                for y_idx in lines_through[x_idx][d_idx]:
                    affected_y.add(y_idx)
                    
            # Save old counts for affected y
            old_counts = {}
            for y_idx in affected_y:
                old_counts[y_idx] = counts[y_idx].copy()
                
            # Apply addition
            for d_idx in range(len(dirs)):
                for y_idx in lines_through[x_idx][d_idx]:
                    counts[y_idx][d_idx] += 1
                    
            # Check validity
            is_valid = True
            for y_idx in affected_y:
                # Check if y_idx is still valid
                # y_idx is in S? (new x_idx is in S, so if y_idx == x_idx, it is in S)
                in_S = (y_idx in S) or (y_idx == x_idx)
                
                ok = False
                for d_idx in range(len(dirs)):
                    if in_S:
                        # Need count == 1
                        if counts[y_idx][d_idx] == 1:
                            ok = True
                            break
                    else:
                        # Need count == 0
                        if counts[y_idx][d_idx] == 0:
                            ok = True
                            break
                if not ok:
                    is_valid = False
                    break
                    
            if is_valid:
                S.add(x_idx)
                # Update valid array? Not strictly necessary if we check on the fly
                # But we need to know which y are invalid for future checks?
                # Actually, we only check affected y.
                # However, if a y becomes invalid, we should not add more points that make it worse?
                # The greedy approach assumes we only add if ALL y remain valid.
                # So if is_valid is True, all y are still valid.
            else:
                # Revert
                for y_idx in affected_y:
                    counts[y_idx] = old_counts[y_idx]
                    
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    # Construct N
    N = [points[i] for i in range(n_points) if i not in best_S]
    
    # Convert to array
    arr = np.array(N, dtype=np.int64)
    return arr

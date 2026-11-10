import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3
    
    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k) % p, (a0*b1 + a1*b0) % p)
        
    def inv(a):
        a0, a1 = a
        norm = (a0*a0 + a1*a1*k) % p
        if norm == 0:
            return None
        norm_inv = pow(norm, p-2, p)
        a_p = (a0, (-a1) % p)
        return mul(a_p, (norm_inv, 0))

    # Generate all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
                    
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    idx_to_point = {i: pt for i, pt in enumerate(all_points)}
    
    # Directions: (1, m) for m in F_q, and (0, 1)
    F_q_elems = [(a, b) for a in range(p) for b in range(p)]
    directions = []
    for m in F_q_elems:
        directions.append(((1, 0), m)) # (1, m)
    directions.append(((0, 1), (0, 0))) # (0, 1)
    
    # Precompute lines for each point and direction
    # lines[pt_idx][dir_idx] = list of point indices on the line
    lines = {}
    for pt_idx, pt in enumerate(all_points):
        lines[pt_idx] = []
        for v in directions:
            line_pts = set()
            for t in F_q_elems:
                tv = mul(t, v)
                x = add(pt, tv)
                line_pts.add(point_to_idx[x])
            lines[pt_idx].append(line_pts)
            
    # Initialize C as Parabola y = x^2
    # Note: y is the second coordinate. x is the first.
    # Point is ((xa, xb), (ya, yb)).
    # We need y = x^2 in F_q.
    # x = (xa, xb). x^2 = mul(x, x).
    C_indices = set()
    for x in F_q_elems:
        y = mul(x, x)
        pt = (x, y)
        C_indices.add(point_to_idx[pt])
        
    # Maintain D_iso and D_clean
    # D_iso[pt_idx] = set of direction indices such that L(pt, d) \cap C == {pt}
    # D_clean[pt_idx] = set of direction indices such that L(pt, d) \cap C == empty
    D_iso = [set() for _ in range(q)]
    D_clean = [set() for _ in range(q)]
    
    def update_directions():
        for i in range(q):
            D_iso[i] = set()
            D_clean[i] = set()
            for d_idx in range(len(directions)):
                line_pts = lines[i][d_idx]
                inter = line_pts & C_indices
                if i in C_indices:
                    if inter == {i}:
                        D_iso[i].add(d_idx)
                else:
                    if inter == set():
                        D_clean[i].add(d_idx)
                        
    update_directions()
    
    # Greedy growth
    while True:
        best_candidate = None
        best_score = -1
        
        for x_idx in range(q):
            if x_idx in C_indices:
                continue
            if not D_clean[x_idx]:
                continue
            
            # Check if x can be added
            valid = True
            
            # Check Condition 1 for existing C
            for y_idx in C_indices:
                # y needs a direction in D_iso[y_idx] that does NOT contain x
                # d contains x if x is in lines[y_idx][d]
                # We need d in D_iso[y_idx] such that x NOT in lines[y_idx][d]
                has_safe_dir = False
                for d_idx in D_iso[y_idx]:
                    if x_idx not in lines[y_idx][d_idx]:
                        has_safe_dir = True
                        break
                if not has_safe_dir:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check Condition 2 for existing non-C
            for z_idx in range(q):
                if z_idx in C_indices or z_idx == x_idx:
                    continue
                if not D_clean[z_idx]:
                    # Should not happen if C is valid
                    continue
                # z needs a direction in D_clean[z_idx] that does NOT contain x
                has_safe_dir = False
                for d_idx in D_clean[z_idx]:
                    if x_idx not in lines[z_idx][d_idx]:
                        has_safe_dir = True
                        break
                if not has_safe_dir:
                    valid = False
                    break
            
            if valid:
                # Score: maybe prefer points that keep more options?
                # Just take the first one or one with max remaining options
                # For simplicity, take first
                best_candidate = x_idx
                break
                
        if best_candidate is None:
            break
            
        C_indices.add(best_candidate)
        update_directions()
        
    # Construct N
    points_in_N = []
    for i, pt in enumerate(all_points):
        if i not in C_indices:
            (xa, xb), (ya, yb) = pt
            points_in_N.append([[xa, xb], [ya, yb]])
            
    return np.array(points_in_N, dtype=np.int64)

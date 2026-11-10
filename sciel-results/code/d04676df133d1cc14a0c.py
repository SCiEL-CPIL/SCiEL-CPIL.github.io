import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # Build field operations
    def field_add(x, y):
        return ((x[0]+y[0])%p, (x[1]+y[1])%p)
    
    def field_mul(x, y):
        return ((x[0]*y[0] + k*x[1]*y[1])%p, (x[0]*y[1]+x[1]*y[0])%p)
    
    def field_neg(x):
        return ((-x[0])%p, (-x[1])%p)
    
    def field_sub(x, y):
        return field_add(x, field_neg(y))
    
    # Enumerate all field elements
    field_elements = [(a,b) for a in range(p) for b in range(p)]
    
    # All points in F_49^2
    all_points = [((xa,xb),(ya,yb)) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    N_pts = len(all_points) # 2401
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Directions: (1,m) for m in F_49, and (0,1)
    # Total 49 + 1 = 50 directions
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1,0), (ma,mb))) # (1, m)
    directions.append(((0,1), (0,0))) # (0, 1)
    
    # Build lines
    # A line is defined by a direction d and a point p on it.
    # We want a canonical representation to avoid duplicates.
    # For direction (1,m), line is y = m*x + c. c is the y-intercept.
    # For direction (0,1), line is x = const.
    
    lines = []
    line_id_for_points = [[None for _ in range(N_pts)] for _ in range(N_pts)]
    points_on_line = []
    
    # Vertical lines (direction (0,1))
    for xa in range(p):
        for xb in range(p):
            line_pts = set()
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa,xb),(ya,yb))
                    line_pts.add(point_to_idx[pt])
            lines.append(line_pts)
            lid = len(lines) - 1
            for pt_idx in line_pts:
                # Store line id in pair (pt_idx, other_pt) later?
                # We need line_id[z1, z2]
                pass
            points_on_line.append(line_pts)
            
    # Slanted lines (direction (1,m))
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = set()
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = field_mul(m, x)
                            y = field_add(mx, c)
                            pt = (x, y)
                            line_pts.add(point_to_idx[pt])
                    lines.append(line_pts)
                    lid = len(lines) - 1
                    points_on_line.append(line_pts)
                    
    # Total lines should be 49 + 2401 = 2450
    assert len(lines) == 2450
    
    # Precompute line_id[z1, z2]
    # For each line, for each pair of points on it, set the ID
    for lid, line_pts in enumerate(lines):
        pts_list = list(line_pts)
        for i in range(len(pts_list)):
            for j in range(i+1, len(pts_list)):
                z1 = pts_list[i]
                z2 = pts_list[j]
                line_id_for_points[z1][z2] = lid
                line_id_for_points[z2][z1] = lid
                
    # Precompute lines_through[z]
    lines_through = [set() for _ in range(N_pts)]
    for lid, line_pts in enumerate(lines):
        for z in line_pts:
            lines_through[z].add(lid)
            
    # Verify each point has 50 lines
    for z in range(N_pts):
        assert len(lines_through[z]) == 50, f"Point {z} has {len(lines_through[z])} lines"
        
    # Initialize state
    C = set()
    good = [set(lines_through[z]) for z in range(N_pts)]
    bad_points = set()
    
    # Greedy loop
    # To improve score, we can try multiple random restarts or different orders
    # But one good run might be enough.
    # Let's use a random order with a fixed seed for reproducibility in local testing,
    # but maybe random seed for evaluation?
    # The task doesn't require determinism, but reproducibility is nice.
    # However, to beat the best score, we might need to try a few strategies.
    
    # Strategy: Random selection from valid candidates.
    # We can also try to pick the candidate that adds the fewest points to bad_points.
    # That requires calculating the delta for each candidate, which is O(N^2) per step.
    # Too slow.
    # Let's stick to random.
    
    random.seed(42)
    
    while True:
        if time.time() - start_time > deadline:
            break
            
        # Find valid candidates
        # A point x is valid if x not in C and x not in bad_points
        # We can maintain a list of available points?
        # Since N is small, just filter.
        candidates = [x for x in range(N_pts) if x not in C and x not in bad_points]
        
        if not candidates:
            break
            
        x = random.choice(candidates)
        C.add(x)
        
        # Update state
        for z in range(N_pts):
            if z == x:
                continue
            L = line_id_for_points[z][x]
            if L in good[z]:
                good[z].remove(L)
                if len(good[z]) == 1:
                    L_crit = next(iter(good[z]))
                    # Add all points on L_crit (except z) to bad_points
                    for w in points_on_line[L_crit]:
                        if w != z:
                            bad_points.add(w)
                elif len(good[z]) == 0:
                    # This should not happen if we only pick valid x
                    # But if it does, we have a bug or the set is invalid
                    pass
        
    # Construct N
    N_points = []
    for i, pt in enumerate(all_points):
        if i not in C:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)

import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3 # alpha^2 = 3
    
    # Field ops
    def add_f(a, b):
        return ((a[0]+b[0])%p, (a[1]+b[1])%p)
    def sub_f(a, b):
        return ((a[0]-b[0])%p, (a[1]-b[1])%p)
    def mul_f(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k)%p, (a0*b1 + a1*b0)%p)
    def neg_f(a):
        return ((-a[0])%p, (-a[1])%p)
    def inv_f(a):
        # a0 + a1 alpha
        # (a0 - a1 alpha) / (a0^2 - a1^2 k)
        denom = (a[0]*a[0] - a[1]*a[1]*k) % p
        if denom == 0:
            return (0,0) # should not happen for non-zero
        d_inv = pow(denom, p-2, p)
        return mul_f((a[0], (-a[1])%p), (d_inv, 0))
        
    # Points in F_49^2
    # Index: (x_a, x_b, y_a, y_b)
    
    # Construct Unital U
    # 2*xa + ya^2 - k*yb^2 = 0
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - k*yb*yb) % p == 0:
                        U.add((xa, xb, ya, yb))
                        
    # All points
    all_points = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.add((xa, xb, ya, yb))
                    
    outside_U = all_points - U
    
    # Lines
    # Direction (1, m) and (0, 1)
    # m in F_49
    lines = []
    # Vertical lines x = c
    for c in [(a,b) for a in range(p) for b in range(p)]:
        pts = []
        for ya in range(p):
            for yb in range(p):
                pts.append((c[0], c[1], ya, yb))
        lines.append(pts)
        
    # Slanted lines y = m*x + b
    for m in [(a,b) for a in range(p) for b in range(p)]:
        for b in [(a,b) for a in range(p) for b in range(p)]:
            pts = []
            for x in [(a,b) for a in range(p) for b in range(p)]:
                # y = m*x + b
                y = add_f(mul_f(m, x), b)
                pts.append((x[0], x[1], y[0], y[1]))
            lines.append(pts)
            
    # For each line, compute intersection with U
    line_U_intersections = []
    for L in lines:
        I = set(pt for pt in L if pt in U)
        line_U_intersections.append(I)
        
    # Greedy Set Cover for outside_U
    # We want to cover all points in outside_U with lines.
    # When we pick a line, we must add its U-intersection to H.
    # We want to minimize |H|.
    
    uncovered = set(outside_U)
    H = set()
    
    # Precompute for each line: which outside points it covers
    line_outside_coverage = []
    for i, L in enumerate(lines):
        cov = set(pt for pt in L if pt in uncovered)
        line_outside_coverage.append(cov)
        
    # Greedy
    while uncovered:
        best_line_idx = -1
        best_score = -1
        # Score: number of new points covered / cost in H
        # Cost is size of (I_L \ H)
        for i in range(len(lines)):
            if not line_outside_coverage[i]:
                continue
            new_cov = line_outside_coverage[i]
            # All points in new_cov are currently uncovered?
            # line_outside_coverage is static, so we need to intersect with uncovered
            actual_new_cov = new_cov & uncovered
            if not actual_new_cov:
                continue
            
            I_L = line_U_intersections[i]
            new_H_cost = len(I_L - H)
            
            # We want to maximize coverage per cost
            # If cost is 0, it's great
            score = len(actual_new_cov) / (new_H_cost + 1)
            
            if score > best_score:
                best_score = score
                best_line_idx = i
                
        if best_line_idx == -1:
            break
            
        # Pick best line
        L = lines[best_line_idx]
        I_L = line_U_intersections[best_line_idx]
        H.update(I_L)
        
        # Update uncovered
        for pt in L:
            if pt in uncovered:
                uncovered.remove(pt)
                
        # Update coverage for all lines? 
        # Or just recompute in next iteration. 
        # Recomputing is O(N_lines * N_pts) which is 2450 * 2400 ~ 6M. 
        # 2000 iterations * 6M = 12B. Too slow.
        
        # Better: maintain uncovered set, and just check intersection.
        # The loop above does `actual_new_cov = new_cov & uncovered`.
        # This is fast if sets are small.
        
    # H is the set of points in U that are in N.
    # C = U - H
    C = U - H
    
    # N = All - C
    N = all_points - C
    
    # Convert to array
    points = []
    for pt in N:
        points.append([[pt[0], pt[1]], [pt[2], pt[3]]])
        
    return np.array(points, dtype=np.int64)

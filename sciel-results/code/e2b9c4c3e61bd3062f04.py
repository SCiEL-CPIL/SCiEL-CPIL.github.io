import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 580
    p = 7
    q = 49
    k = 3
    
    # Precompute field operations
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    
    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            add_tbl[i, j] = ((a0+b0)%p)*p + ((a1+b1)%p)
            sub_tbl[i, j] = ((a0-b0)%p)*p + ((a1-b1)%p)
            mul_tbl[i, j] = ((a0*b0+a1*b1*k)%p)*p + ((a0*b1+a1*b0)%p)
            
    total_points = q * q
    
    # Unital membership
    U_arr = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - k*yb*yb) % p == 0:
                        U_arr[(xa*p+xb)*q + (ya*p+yb)] = True
                        
    U_set = set(np.where(U_arr)[0].tolist())
    all_pts = set(range(total_points))
    
    # Build lines
    lines = []
    line_points = [] # List of lists of point indices
    for c in range(q):
        pts = [c*q+y for y in range(q)]
        lines.append(pts)
        line_points.append(pts)
        
    for m in range(q):
        for b in range(q):
            pts = [x*q + add_tbl[mul_tbl[m,x],b] for x in range(q)]
            lines.append(pts)
            line_points.append(pts)
            
    num_lines = len(lines)
    
    # Precompute lines_through_point
    lines_through_point = [[] for _ in range(total_points)]
    for i, pts in enumerate(line_points):
        for pt in pts:
            lines_through_point[pt].append(i)
            
    best_H = set(U_set)
    best_size = len(best_H)
    
    for attempt in range(20):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        rng = random.Random(attempt * 12345 + 67890)
        H = set(U_set)
        
        # valid_lines[p] is a set of line indices that are valid for p
        # A line L is valid for p if L cap U is subset of H
        # Initially H=U, so all lines are valid.
        valid_lines = [set(lines_through_point[p]) for p in range(total_points)]
        
        order = list(U_set)
        rng.shuffle(order)
        
        for x in order:
            affected_lines = lines_through_point[x]
            affected_points = set()
            for li in affected_lines:
                affected_points.update(line_points[li])
                
            # Tentatively remove x from H
            # This makes all affected_lines invalid for all affected_points
            # Check if any affected_point loses all valid lines
            
            for pt in affected_points:
                valid_lines[pt].difference_update(affected_lines)
                
            is_valid = True
            for pt in affected_points:
                if len(valid_lines[pt]) == 0:
                    is_valid = False
                    break
                    
            if not is_valid:
                # Restore
                for pt in affected_points:
                    valid_lines[pt].update(affected_lines)
            else:
                # Keep x removed
                H.remove(x)
                
        if len(H) < best_size:
            best_size = len(H)
            best_H = set(H)
            
    # Construct N
    C = U_set - best_H
    N = all_pts - C
    
    points = []
    for pt in sorted(N):
        x = pt // q
        y = pt % q
        xa, xb = x // p, x % p
        ya, yb = y // p, y % p
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

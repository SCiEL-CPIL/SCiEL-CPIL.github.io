import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    p = 7
    q = p * p
    k = 3

    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    # Generate all points
    # Index: x = (xa, xb), y = (ya, yb)
    # idx = ((xa * p + xb) * p + ya) * p + yb
    def point_to_index(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def index_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)

    total_points = q * q
    all_points = list(range(total_points))
    
    # Generate lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    # m = (ma, mb)
    
    lines = [] # list of sets of point indices
    point_to_lines = {i: [] for i in range(total_points)}
    
    # Helper to add a line
    def add_line(points_on_line):
        line_set = set(points_on_line)
        lines.append(line_set)
        line_id = len(lines) - 1
        for p_idx in line_set:
            point_to_lines[p_idx].append(line_id)
            
    # Direction (0, 1): x = c
    for ca in range(p):
        for cb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = point_to_index(ca, cb, ya, yb)
                    pts.append(idx)
            add_line(pts)
            
    # Direction (1, m): y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = point_to_index(xa, xb, y[0], y[1])
                            pts.append(idx)
                    add_line(pts)
                    
    # Greedy Algorithm
    S = set()
    line_counts = np.zeros(len(lines), dtype=np.int32)
    witnesses = {i: set() for i in range(total_points)}
    
    # Order of points
    # Randomize to avoid bias
    random.seed(42)
    point_order = all_points[:]
    random.shuffle(point_order)
    
    # We can try multiple restarts if time permits
    best_S = set()
    
    for restart in range(10): # Try a few restarts
        if time.time() > deadline - 60:
            break
            
        S = set()
        line_counts = np.zeros(len(lines), dtype=np.int32)
        witnesses = {i: set() for i in range(total_points)}
        
        # Shuffle again for each restart
        random.shuffle(point_order)
        
        for x_idx in point_order:
            if time.time() > deadline - 30:
                break
                
            lines_through_x = point_to_lines[x_idx]
            
            # Check if x can be added: needs a line with count 0
            has_empty_line = False
            for L in lines_through_x:
                if line_counts[L] == 0:
                    has_empty_line = True
                    break
            
            if not has_empty_line:
                continue
                
            # Check if adding x invalidates any y in S
            valid = True
            # We need to check if Witnesses(y) is subset of lines_through_x
            # Optimization: If Witnesses(y) is empty, it's already invalid (shouldn't happen)
            # If Witnesses(y) has an element NOT in lines_through_x, it's safe.
            
            # To speed up, we can precompute a bitmask or use sets.
            # Sets are fine for 50 elements.
            lines_through_x_set = set(lines_through_x)
            
            for y_idx in S:
                W_y = witnesses[y_idx]
                if not W_y:
                    valid = False
                    break
                # Check if W_y is subset of lines_through_x_set
                # If any element in W_y is not in lines_through_x_set, safe.
                is_subset = True
                for L in W_y:
                    if L not in lines_through_x_set:
                        is_subset = False
                        break
                if is_subset:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Add x
            S.add(x_idx)
            
            # Update line_counts
            for L in lines_through_x:
                line_counts[L] += 1
                
            # Update witnesses for existing y
            for y_idx in S:
                if y_idx == x_idx:
                    continue
                W_y = witnesses[y_idx]
                # Remove lines that now have count > 1?
                # A line L is a witness for y if line_counts[L] == 1.
                # Before adding x, line_counts[L] was 1 for L in W_y.
                # After adding x, if x is on L, line_counts[L] becomes 2.
                # So we remove L from W_y if x is on L.
                # We already checked that W_y is not subset of lines_through_x.
                # So we just remove the intersection.
                intersection = W_y.intersection(lines_through_x_set)
                W_y.difference_update(intersection)
                
            # Set witnesses for x
            # Witness lines for x are those L in lines_through_x with line_counts[L] == 1 (which were 0 before)
            W_x = set()
            for L in lines_through_x:
                if line_counts[L] == 1:
                    W_x.add(L)
            witnesses[x_idx] = W_x
            
        if len(S) > len(best_S):
            best_S = set(S)
            
    # Construct output
    # N = F^2 \ S
    # Return N as array of shape (k, 2, 2)
    
    if not best_S:
        # Fallback
        return np.array([[[0, 0], [0, 0]]], dtype=np.int64)
        
    points = []
    for idx in range(total_points):
        if idx not in best_S:
            xa, xb, ya, yb = index_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600
    p = 7
    q = 49
    k = 3
    
    # Field operations
    def add_f(a, b):
        return ((a[0]+b[0])%p, (a[1]+b[1])%p)
    def mul_f(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k)%p, (a0*b1 + a1*b0)%p)
        
    # Points in F_49^2
    # Index: (x_a, x_b, y_a, y_b)
    all_points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = (xa, xb, ya, yb)
                    idx = len(all_points)
                    all_points.append(pt)
                    point_to_idx[pt] = idx
                    
    total_points = len(all_points)
    
    # Identify Unital U
    # 2*xa + ya^2 - k*yb^2 = 0
    U_set = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - k*yb*yb) % p == 0:
                        U_set.add((xa, xb, ya, yb))
                        
    U_indices = [point_to_idx[pt] for pt in U_set]
    U_bool = np.zeros(total_points, dtype=bool)
    for idx in U_indices:
        U_bool[idx] = True
        
    # Generate Lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    lines = []
    point_to_lines = [[] for _ in range(total_points)]
    
    # Vertical lines (direction (0,1))
    for c in range(q):
        c_a, c_b = c // p, c % p
        line_pts = []
        for ya in range(p):
            for yb in range(p):
                pt = (c_a, c_b, ya, yb)
                line_pts.append(point_to_idx[pt])
        lines.append(line_pts)
        
    # Slanted lines (direction (1, m))
    for m in range(q):
        m_a, m_b = m // p, m % p
        for b in range(q):
            b_a, b_b = b // p, b % p
            line_pts = []
            for x in range(q):
                x_a, x_b = x // p, x % p
                # y = m*x + b
                mx = mul_f((m_a, m_b), (x_a, x_b))
                y = add_f(mx, (b_a, b_b))
                pt = (x_a, x_b, y[0], y[1])
                line_pts.append(point_to_idx[pt])
            lines.append(line_pts)
            
    num_lines = len(lines)
    
    # Precompute line-U intersections and point-line mappings
    line_U_sets = []
    for L in lines:
        u_pts = [pt for pt in L if U_bool[pt]]
        line_U_sets.append(set(u_pts))
        
    # Map point to lines passing through it
    for lid, L in enumerate(lines):
        for pt in L:
            point_to_lines[pt].append(lid)
            
    # Helper: Check if S is valid
    # S is a set of indices
    def is_valid(S_set):
        # 1. For every x in S: exists L through x with L ∩ S == {x}
        for x in S_set:
            found = False
            for lid in point_to_lines[x]:
                # Check if L ∩ S == {x}
                # We know x is in L.
                # We need to check if any other point in S is in L.
                # L ∩ S = {y in S | y in L}
                # Efficient check:
                # We can precompute for each line L, the set of points in S?
                # No, S changes.
                # But |S| is small (~300).
                # For a given x and L, check if any y in S (y!=x) is in L.
                # This is O(|S|) per line.
                # Total O(|S| * 50 * |S|) = 300 * 50 * 300 = 4.5M.
                # Might be slow in Python if called many times.
                
                # Optimization:
                # For each line L, maintain a count of S-points?
                # We are doing this in a check function, so we can't easily maintain global state without passing it.
                # Let's just do the direct check but optimized.
                
                # Actually, we can check:
                # Is L ∩ S a subset of {x}?
                # i.e., is L ∩ (S \ {x}) empty?
                
                # To speed up, we can precompute for each line L, the list of points.
                # And check membership.
                
                # Let's try a different approach for the check:
                # For each line L, let count_L = |L ∩ S|.
                # If we maintain these counts, we can check in O(1) per line.
                # But we are in a pure function `is_valid`.
                
                # Let's assume the number of candidates is small enough that O(|S|^2 * 50) is okay for a few thousand checks.
                # If we do 1000 checks, 4.5B ops is too slow.
                
                # We need to maintain counts.
                pass
            
            # This naive check is too slow for iterative improvement.
            # We will implement the main loop with maintained counts.
            return False # Placeholder
            
    # Main Logic: Remove then Add
    
    # Initial S = U
    S = set(U_indices)
    
    # Precompute line counts for S
    line_counts = np.zeros(num_lines, dtype=int)
    for idx in S:
        for lid in point_to_lines[idx]:
            line_counts[lid] += 1
            
    # Helper to check validity given S and line_counts
    def check_validity(S_set, l_counts):
        # Condition 1: For x in S, exists L through x with l_counts[L] == 1
        for x in S_set:
            ok = False
            for lid in point_to_lines[x]:
                if l_counts[lid] == 1:
                    ok = True
                    break
            if not ok:
                return False
                
        # Condition 2: For z not in S, exists L through z with l_counts[L] == 0
        # We only need to check z not in S.
        # To avoid iterating all 2401 points every time, we can just iterate all and skip if in S.
        # 2401 * 50 = 120k checks. Fast.
        for z in range(total_points):
            if z in S_set:
                continue
            ok = False
            for lid in point_to_lines[z]:
                if l_counts[lid] == 0:
                    ok = True
                    break
            if not ok:
                return False
        return True
        
    # Phase 1: Remove until stable
    changed = True
    while changed:
        changed = False
        # Find a point in S with no private line
        bad_point = -1
        for x in S:
            ok = False
            for lid in point_to_lines[x]:
                if line_counts[lid] == 1:
                    ok = True
                    break
            if not ok:
                bad_point = x
                break
                
        if bad_point != -1:
            # Remove bad_point
            S.remove(bad_point)
            for lid in point_to_lines[bad_point]:
                line_counts[lid] -= 1
            changed = True
            
    # Phase 2: Greedy Add
    # Candidates are points in U \ S
    candidates = [idx for idx in U_indices if idx not in S]
    
    # Sort candidates by some heuristic?
    # Maybe points with many lines that are currently "empty" (count 0) or "singleton" (count 1)?
    # If we add x, we need a line with count 0 (becomes 1) or count 1 (becomes 2, bad for x if we need count 1?
    # Wait, if x is added, we need a line L through x with L ∩ S_new = {x}.
    # This means L ∩ S_old must be empty.
    # So x must have at least one line with count 0 in the current state.
    
    # So we can filter candidates:
    valid_candidates = []
    for idx in candidates:
        has_empty_line = False
        for lid in point_to_lines[idx]:
            if line_counts[lid] == 0:
                has_empty_line = True
                break
        if has_empty_line:
            valid_candidates.append(idx)
            
    # Shuffle to avoid bias?
    random.seed(42)
    random.shuffle(valid_candidates)
    
    for idx in valid_candidates:
        if time.time() - start_time > DEADLINE - 60:
            break
            
        # Try adding idx
        S.add(idx)
        for lid in point_to_lines[idx]:
            line_counts[lid] += 1
            
        if check_validity(S, line_counts):
            # Keep
            pass
        else:
            # Revert
            S.remove(idx)
            for lid in point_to_lines[idx]:
                line_counts[lid] -= 1
            
    # Construct N
    N_points = []
    for pt, idx in point_to_idx.items():
        if idx not in S:
            xa, xb, ya, yb = pt
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)

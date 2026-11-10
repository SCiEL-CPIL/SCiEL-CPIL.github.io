import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) = a + b*alpha
    # alpha^2 = 3
    
    # Precompute field operations if needed, but we can work with indices 0..2400
    # Map (xa, xb, ya, yb) to index
    # index = ((xa * p + xb) * p + ya) * p + yb
    
    # Directions: (1, m) for m in F_49, and (0, 1)
    # Total q+1 = 50 directions.
    # We need to map each direction to an index 0..49 for the bitmask.
    
    # Let's represent points as indices 0..2400.
    # point_to_coords: index -> (xa, xb, ya, yb)
    # coords_to_point: (xa, xb, ya, yb) -> index
    
    coords_to_point = {}
    point_to_coords = []
    idx = 0
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    coords_to_point[(xa, xb, ya, yb)] = idx
                    point_to_coords.append((xa, xb, ya, yb))
                    idx += 1
    
    # Define lines
    # A line is defined by a point and a direction.
    # Directions:
    # 0..48: (1, m) where m is 0..48 (representing elements of F_49)
    # 49: (0, 1)
    
    # We need to compute the set of points on each line.
    # There are 2450 lines.
    # lines: list of lists of point indices.
    # line_dir: list of direction indices for each line.
    # point_to_lines: for each point, list of (line_index, direction_index)
    
    # To speed up, we can generate lines by iterating through all points and directions?
    # No, that creates duplicates. A line is unique.
    # Better: iterate through all possible lines.
    # Vertical lines: x = c. 49 lines.
    # Sloped lines: y = m*x + c. For each m (49 values), 49 lines. 49*49 = 2401 lines.
    # Total 2450 lines.
    
    lines = []
    line_dirs = []
    point_to_lines = [[] for _ in range(q*q)]
    
    # Helper to add a point to a line
    def add_point_to_line(line_idx, pt_idx, dir_idx):
        lines[line_idx].append(pt_idx)
        point_to_lines[pt_idx].append((line_idx, dir_idx))
    
    line_idx = 0
    # Vertical lines: x = c, direction (0, 1) -> dir_idx 49
    for c in range(q):
        lines.append([])
        line_dirs.append(49)
        for y in range(q):
            # Point (c, y)
            # c = (ca, cb), y = (ya, yb)
            # We need to iterate y in F_49.
            # Let's just iterate all points and check if x matches c?
            # That's O(q^4) which is 5.7M. Acceptable.
            pass
        
        # More efficient: generate points directly
        # c is an element of F_49. y is an element of F_49.
        # We have coords_to_point.
        # c = (ca, cb).
        ca, cb = divmod(c, p) # Wait, c is 0..48. c = a + b*alpha.
        # My mapping: index = ((xa * p + xb) * p + ya) * p + yb
        # So xa = (idx // (p*p*p)) % p?
        # Let's stick to the nested loops for generation to be safe.
        
        # Reset
        lines.pop()
        line_dirs.pop()
        
        for y_idx in range(q):
            # Get y coords
            ya, yb = divmod(y_idx, p)
            # Point (ca, cb, ya, yb)
            pt_idx = coords_to_point[(ca, cb, ya, yb)]
            add_point_to_line(line_idx, pt_idx, 49)
        line_idx += 1
        
    # Sloped lines: y = m*x + c
    # m in F_49 (0..48), c in F_49 (0..48)
    for m_idx in range(q):
        m_a, m_b = divmod(m_idx, p)
        for c_idx in range(q):
            c_a, c_b = divmod(c_idx, p)
            lines.append([])
            line_dirs.append(m_idx) # dir index 0..48 corresponds to m
            # Iterate x in F_49
            for x_idx in range(q):
                x_a, x_b = divmod(x_idx, p)
                # y = m*x + c
                # (ma + mb*alpha) * (xa + xb*alpha) + (ca + cb*alpha)
                # = ma*xa + ma*xb*alpha + mb*alpha*xa + mb*xb*alpha^2 + ca + cb*alpha
                # alpha^2 = 3
                # = (ma*xa + 3*mb*xb + ca) + (ma*xb + mb*xa + cb)*alpha
                y_a = (m_a * x_a + 3 * m_b * x_b + c_a) % p
                y_b = (m_a * x_b + m_b * x_a + c_b) % p
                pt_idx = coords_to_point[(x_a, x_b, y_a, y_b)]
                add_point_to_line(line_idx, pt_idx, m_idx)
            line_idx += 1
            
    # Verification: 49 + 49*49 = 2450 lines.
    
    # Greedy Algorithm with Random Restarts
    start_time = time.time()
    time_limit = 600 # seconds
    safety_margin = 30
    
    best_S_size = 0
    best_S_mask = np.zeros(q*q, dtype=bool)
    
    # Precompute for each line, the list of points
    # And for each point, the list of lines it belongs to, with direction index
    
    # We will try to add points to S greedily.
    # Order: random permutation of points.
    
    num_restarts = 0
    while time.time() - start_time < time_limit - safety_margin:
        num_restarts += 1
        # Random order
        order = np.random.permutation(q*q)
        
        S_mask = np.zeros(q*q, dtype=bool)
        # safe_mask[x]: bitmask of directions that are safe.
        # Initially all 50 directions are safe.
        # 50 bits fit in a uint64.
        safe_mask = np.full(q*q, (1 << 50) - 1, dtype=np.uint64)
        
        # s_count_on_line[L]: number of points in L that are in S.
        s_count_on_line = np.zeros(len(lines), dtype=np.int32)
        
        current_size = 0
        valid = True
        
        for pt in order:
            # Try to add pt to S
            # Check if it causes any safe_mask to become 0
            
            # 1. Update s_count_on_line for lines containing pt
            affected_lines = []
            for line_idx, dir_idx in point_to_lines[pt]:
                affected_lines.append((line_idx, dir_idx))
                s_count_on_line[line_idx] += 1
                
            # 2. Check validity for pt itself
            # For each direction d (line L through pt):
            # L \ {pt} subset S_new?
            # S_new contains pt.
            # L \ {pt} subset S_old U {pt} -> L \ {pt} subset S_old (since pt not in L\{pt})
            # So check if s_count_on_line[L]_old == q - 1.
            # Note: s_count_on_line[L] is now incremented.
            # So we need s_count_on_line[L] - 1 == q - 1 => s_count_on_line[L] == q.
            
            for line_idx, dir_idx in affected_lines:
                if s_count_on_line[line_idx] == q:
                    # Direction dir_idx is unsafe for pt
                    safe_mask[pt] &= ~(1 << dir_idx)
                    if safe_mask[pt] == 0:
                        # Invalid
                        for l_idx, _ in affected_lines:
                            s_count_on_line[l_idx] -= 1
                        valid = False
                        break
            
            if not valid:
                continue
                
            # 3. Check validity for other points on these lines
            for line_idx, dir_idx in affected_lines:
                L = lines[line_idx]
                # For each x in L \ {pt}
                for x in L:
                    if x == pt:
                        continue
                    # Check if L \ {x} subset S_new
                    # S_new contains pt.
                    # L \ {x} = (L \ {pt, x}) U {pt}
                    # So L \ {x} subset S_new iff (L \ {pt, x}) subset S_old
                    # s_count_on_line[L]_new = s_count_on_line[L]_old + 1
                    # s_count_on_line[L]_old - (1 if x in S_old else 0) == q - 2
                    # Note: x is not in S_old (we are adding pt, x was already in S or not)
                    # Wait, x might be in S_old.
                    # If x in S_old, then x is counted in s_count_on_line[L]_old.
                    # L \ {pt, x} size is q - 2.
                    # We need all q-2 points to be in S_old.
                    # So s_count_on_line[L]_old - (1 if x in S_old else 0) == q - 2.
                    
                    if S_mask[x]:
                        count_excl_x = s_count_on_line[line_idx] - 1 - 1 # -1 for pt, -1 for x
                    else:
                        count_excl_x = s_count_on_line[line_idx] - 1 # -1 for pt
                    
                    if count_excl_x == q - 2:
                        # Direction dir_idx is unsafe for x
                        safe_mask[x] &= ~(1 << dir_idx)
                        if safe_mask[x] == 0:
                            # Invalid
                            for l_idx, _ in affected_lines:
                                s_count_on_line[l_idx] -= 1
                            valid = False
                            break
                            
                if not valid:
                    break
                    
            if valid:
                S_mask[pt] = True
                current_size += 1
            else:
                # Revert s_count changes
                for l_idx, _ in affected_lines:
                    s_count_on_line[l_idx] -= 1
                
        if current_size > best_S_size:
            best_S_size = current_size
            best_S_mask = S_mask.copy()
            # Early exit if we found a large set?
            # Max possible size?
            # If we get close to theoretical max, we can stop.
            # Theoretical max is not known, but current best is ~1000+.
            
    # Construct N from best_S_mask
    # N = complement of S
    points = []
    for i in range(q*q):
        if not best_S_mask[i]:
            xa, xb, ya, yb = point_to_coords[i]
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

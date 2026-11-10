# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Precompute lines and incidence.
    2. Use a fast greedy algorithm with multiple restarts to build a large set S (complement of N) 
       such that every point x in F_49^2 has at least one line through x that is "S-clean" 
       (contains no points of S other than possibly x).
    3. The condition for x to be satisfied is that there exists a line L through x 
       such that (L \\ {x}) is disjoint from S.
    
    We maximize |S| to maximize the score (which is proportional to 2401 - |N| = |S|).
    """
    p_prime = 7
    q = 49
    N_points = q * q  # 2401
    
    # --- Field Arithmetic F_49 ---
    # Represent elements as (a, b) where value is a + b*alpha, alpha^2 = 3.
    # idx = a * 7 + b.
    
    def add_idx(i, j):
        a1, b1 = divmod(i, 7)
        a2, b2 = divmod(j, 7)
        return ((a1 + a2) % 7) * 7 + (b1 + b2) % 7

    def mul_idx(i, j):
        a1, b1 = divmod(i, 7)
        a2, b2 = divmod(j, 7)
        # (a1 + b1*a) * (a2 + b2*a) = a1a2 + (a1b2 + b1a2)a + b1b2 a^2
        # a^2 = 3
        real = (a1 * a2 + 3 * b1 * b2) % 7
        imag = (a1 * b2 + b1 * a2) % 7
        return real * 7 + imag

    def inv_idx(i):
        # i = a + b*a. Norm = a^2 - 3b^2. Inv = (a - b*a) / Norm.
        a, b = divmod(i, 7)
        norm = (a * a - 3 * b * b) % 7
        if norm == 0:
            return 0 # Should not happen for non-zero i in field
        # Inverse of norm in F_7
        norm_inv = pow(norm, -1, 7) if norm != 0 else 0
        real = (a * norm_inv) % 7
        imag = (6 * b * norm_inv) % 7  # -b
        return real * 7 + imag

    # Precompute addition and multiplication tables for speed?
    # 49x49 tables are small.
    add_table = np.zeros((49, 49), dtype=np.int32)
    mul_table = np.zeros((49, 49), dtype=np.int32)
    inv_table = np.zeros(49, dtype=np.int32)
    
    for i in range(49):
        for j in range(49):
            add_table[i, j] = add_idx(i, j)
            mul_table[i, j] = mul_idx(i, j)
        if i != 0:
            inv_table[i] = inv_idx(i)
        else:
            inv_table[i] = 0
            
    # --- Point Representation ---
    # Point P = (x, y) where x, y in F_49.
    # Index in 2D array: p_idx = x * 49 + y.
    
    # --- Line Representation ---
    # Lines are defined by slope m in F_49 (49 slopes) and vertical lines.
    # Total 50 lines through any point.
    # We represent a line by (type, param).
    # Type 0: Vertical. Param x. Line: {(x, y) for y in F_49}.
    # Type 1: Slope m. Param c. Line: {(x, y) for x in F_49, y = m*x + c}.
    
    # We need to iterate through all lines efficiently.
    # There are 50 * 49 = 2450 lines.
    
    # For the greedy algorithm, we need to know for each point p, 
    # which lines pass through it.
    # And for each line L, which points are on it.
    
    # Let's precompute the points on each line.
    # lines: list of 2450 arrays of 49 point indices.
    
    lines = []
    line_points = [] # list of lists of point indices
    
    # Vertical lines
    for x in range(49):
        pts = []
        for y in range(49):
            pts.append(x * 49 + y)
        line_points.append(np.array(pts, dtype=np.int32))
        
    # Sloped lines
    for m in range(49):
        for c in range(49):
            pts = []
            for x in range(49):
                # y = m*x + c
                y = add_idx(mul_table[m, x], c)
                pts.append(x * 49 + y)
            line_points.append(np.array(pts, dtype=np.int32))
            
    # Precompute: for each point, list of line indices passing through it.
    point_lines = [[] for _ in range(2401)]
    for l_idx, pts in enumerate(line_points):
        for p in pts:
            point_lines[p].append(l_idx)
            
    # Convert to numpy arrays for faster access? 
    # List of lists is probably fine if we just iterate.
    # But checking "is line disjoint from S" requires checking if any point in line is in S.
    
    # --- Greedy Algorithm ---
    
    def greedy_build(seed):
        rng = np.random.RandomState(seed)
        # Order of points to consider adding to S
        order = np.arange(2401)
        rng.shuffle(order)
        
        in_S = np.zeros(2401, dtype=bool)
        S_count_on_line = np.zeros(2450, dtype=np.int32)
        valid_lines_count = np.full(2401, 50, dtype=np.int32)
        
        for p_idx in order:
            # Check if adding p_idx is possible
            # We need to see if valid_lines_count[x] > 0 for all x after update.
            
            # Collect affected points
            affected_points = set()
            for L in point_lines[p_idx]:
                for x in line_points[L]:
                    affected_points.add(x)
                    
            # Let's just update valid_lines_count in place and keep a list of (x, delta) to revert.
            changes = []
            
            # Save old counts for lines
            old_counts = []
            for L in point_lines[p_idx]:
                old_counts.append(S_count_on_line[L])
                
            for i, L in enumerate(point_lines[p_idx]):
                old_c = old_counts[i]
                new_c = old_c + 1
                S_count_on_line[L] = new_c
                
                for x in line_points[L]:
                    # Determine old validity of L for x
                    # old_in_S_x = in_S[x] (before update, so p is False)
                    # new_in_S_x = in_S[x] (after update, so p is True)
                    
                    old_in_s = in_S[x]
                    new_in_s = in_S[x]
                    if x == p_idx:
                        new_in_s = True
                        # old_in_s is False (since we haven't updated in_S array yet)
                    
                    # Old validity:
                    if old_in_s:
                        old_valid = (old_c == 1)
                    else:
                        old_valid = (old_c == 0)
                        
                    # New validity:
                    if new_in_s:
                        new_valid = (new_c == 1)
                    else:
                        new_valid = (new_c == 0)
                        
                    if old_valid and not new_valid:
                        valid_lines_count[x] -= 1
                        changes.append((x, -1))
                    elif not old_valid and new_valid:
                        valid_lines_count[x] += 1
                        changes.append((x, 1))
                        
            # Check if any point has valid_lines_count == 0
            # We only need to check affected points?
            # No, if a point was 1 and we decremented it, it becomes 0.
            # If a point was not affected, it remains > 0 (assuming we started valid).
            # So we only need to check affected points.
            
            valid = True
            for x in affected_points:
                if valid_lines_count[x] == 0:
                    valid = False
                    break
            
            if valid:
                in_S[p_idx] = True
                # Commit: changes are already applied to valid_lines_count
                # S_count_on_line is updated
                # in_S is updated
            else:
                # Revert
                for x, delta in changes:
                    valid_lines_count[x] -= delta
                for i, L in enumerate(point_lines[p_idx]):
                    S_count_on_line[L] = old_counts[i]
                # in_S[p_idx] remains False
        
        return in_S

    # --- Main Execution ---
    start_time = time.time()
    deadline = start_time + 580
    
    best_in_S = None
    best_size = 0
    seed_id = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        in_S = greedy_build(seed_id)
        size = np.sum(in_S)
        if size > best_size:
            best_size = size
            best_in_S = in_S
            # Print progress? No, stdout might be captured or slow.
            # Just update.
        seed_id += 1
        
        # If we found a very large set, maybe stop?
        # Max possible S is roughly 2401 * (1/50) * something?
        # Actually, if S is large, it's hard to satisfy conditions.
        # Let's just keep going.
        
    if best_in_S is None:
        # Fallback
        best_in_S = np.zeros(2401, dtype=bool)
        best_in_S[0] = True
        
    # Construct result N
    # N is points NOT in S.
    result_list = []
    for i in range(2401):
        if not best_in_S[i]:
            x = i // 49
            y = i % 49
            xa, xb = divmod(x, 7)
            ya, yb = divmod(y, 7)
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
            
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END

import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    
    # Field operations
    k = 3
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    # Precompute line IDs for each point and direction
    # Direction v = (1, m) for m in F_49, and v = (0, 1)
    # For v = (1, m), line_id = y - m*x
    # For v = (0, 1), line_id = x
    
    # Encode point (xa, xb, ya, yb) to index
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    # Precompute line_id[point_idx, dir_idx]
    # dir_idx 0..48: m = (ma, mb), ma = dir_idx // p, mb = dir_idx % p
    # dir_idx 49: vertical
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = encode(xa, xb, ya, yb)
                    x = (xa, xb)
                    y = (ya, yb)
                    for ma in range(p):
                        for mb in range(p):
                            d_idx = ma * p + mb
                            m = (ma, mb)
                            c = field_add(y, field_neg(field_mul(m, x)))
                            line_id[idx, d_idx] = c[0] * p + c[1]
                    # Vertical direction
                    line_id[idx, n_dirs - 1] = xa * p + xb

    # Validation function
    def is_valid(S_flat):
        """
        S_flat: boolean array of size n_points
        Returns True if S is a valid complement set.
        Condition: For every point y, there exists a direction v such that:
          - if y in S: count of S on L_v(y) is 1
          - if y not in S: count of S on L_v(y) is 0
        """
        satisfied = np.zeros(n_points, dtype=bool)
        
        for d in range(n_dirs):
            # Count S points on each line for direction d
            # Line IDs range from 0 to p*p - 1
            counts = np.zeros(p * p, dtype=np.int32)
            # Use bincount for speed
            lids = line_id[:, d]
            # Only count points in S
            s_indices = np.where(S_flat)[0]
            if len(s_indices) > 0:
                s_lids = lids[s_indices]
                counts = np.bincount(s_lids, minlength=p * p)
            
            # Determine satisfied points for this direction
            # For each point y, check condition based on S_flat[y]
            # If S_flat[y] is True, we need counts[lid] == 1
            # If S_flat[y] is False, we need counts[lid] == 0
            
            # Vectorized check:
            # counts_for_points = counts[line_id[:, d]]
            # valid_for_d = (S_flat & (counts_for_points == 1)) | (~S_flat & (counts_for_points == 0))
            
            counts_for_points = counts[line_id[:, d]]
            valid_for_d = (S_flat & (counts_for_points == 1)) | (~S_flat & (counts_for_points == 0))
            
            satisfied |= valid_for_d
            
            # Early exit if all satisfied
            if np.all(satisfied):
                return True
                
        return False

    # Initial S: Empty
    S = np.zeros(n_points, dtype=bool)
    
    # Greedy construction
    # We want to maximize |S|.
    # Start with empty S, try to add points.
    
    # To improve, we can try multiple random orders.
    # Also, we can try to start with a small random S.
    
    best_S = S.copy()
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    # Precompute list of all points
    all_points = list(range(n_points))
    
    # Try several random orders
    for trial in range(100):
        if time.time() > deadline:
            break
            
        random.shuffle(all_points)
        current_S = np.zeros(n_points, dtype=bool)
        
        for idx in all_points:
            if time.time() > deadline:
                break
                
            # Try adding idx
            current_S[idx] = True
            
            if is_valid(current_S):
                # Keep it
                if np.sum(current_S) > best_size:
                    best_S = current_S.copy()
                    best_size = np.sum(current_S)
            else:
                # Reject
                current_S[idx] = False
                
    # Local search: try to swap or add
    # Since greedy might get stuck, try random restarts with different initial S
    # Or, try to add points that were rejected in the last pass
    
    # Let's do a few more passes on the best_S found so far
    # Try to add any point not in best_S
    for trial in range(50):
        if time.time() > deadline:
            break
            
        current_S = best_S.copy()
        changed = False
        candidates = [i for i in range(n_points) if not current_S[i]]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline:
                break
            current_S[idx] = True
            if is_valid(current_S):
                changed = True
                best_S = current_S.copy()
                best_size = np.sum(current_S)
            else:
                current_S[idx] = False
                
        if not changed:
            break
            
    # Convert best_S to N (complement)
    N_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = encode(xa, xb, ya, yb)
                    if not best_S[idx]:
                        N_points.append([[xa, xb], [ya, yb]])
                        
    if not N_points:
        # Fallback: return a known valid set (e.g., all points)
        # If S is empty, N is all points. Is N=all points valid?
        # If N=all points, S=empty.
        # For any x, line L_v(x) contains 0 points from S.
        # Condition: if x not in S (true), need count 0. Yes.
        # So N=all points is valid.
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)

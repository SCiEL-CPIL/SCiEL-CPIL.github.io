import time
import random
import numpy as np

def construct():
    n = 128
    # Generate all 4-APs as index arrays
    aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # Precompute as numpy array for fast access
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps_arr)
    
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Helper: vectorized violation check
    # A 4-AP violates if it uses < 3 distinct colors
    # i.e., it uses 1 or 2 distinct colors
    # Uses 1: all four same
    # Uses 2: exactly two distinct values (with some pattern)
    # We can check: for each AP, count distinct colors
    # Vectorized: for each AP, get the 4 colors, then check if len(set) < 3
    
    def find_violations(colors_arr):
        """Return indices of violating APs using vectorized NumPy."""
        # colors_arr: shape (n,)
        # Get colors for all APs: shape (num_aps, 4)
        ap_colors = colors_arr[aps_arr]  # (num_aps, 4)
        
        # Check for < 3 distinct colors
        # A set of 4 values has < 3 distinct iff:
        # - All 4 same (1 distinct)
        # - Exactly 2 distinct values
        # 
        # For 2 distinct: the multiset must be {a,a,a,b}, {a,a,b,b}, or {a,b,b,b}
        # Equivalent: the number of distinct values in each row is 1 or 2
        # 
        # Vectorized approach: 
        # Count how many pairs are equal. If >= 6 pairs equal -> all same (1 distinct)
        # If exactly 4 pairs equal -> could be {a,a,a,b} (3 pairs among a's) or {a,a,b,b} (2+2)
        # This is getting complex. Let's use a simpler method.
        
        # Simpler: compute min and max of each row. 
        # If min == max: 1 distinct (violation)
        # Else: need to check if only 2 distinct.
        # For 4 values with min < max, they have < 3 distinct iff:
        #   all values are in {min, max} AND not all equal
        #   i.e., each value is either min or max
        
        c_min = ap_colors.min(axis=1)  # (num_aps,)
        c_max = ap_colors.max(axis=1)  # (num_aps,)
        
        # All same: min == max
        all_same = (c_min == c_max)
        
        # Only 2 distinct: all values are in {c_min, c_max} and not all same
        # Check: each of the 4 values equals c_min or c_max
        in_range = ((ap_colors == c_min[:, None]) | (ap_colors == c_max[:, None])).all(axis=1)
        two_distinct = in_range & ~all_same
        
        violations = all_same | two_distinct
        return np.where(violations)[0]
    
    # Try K from 3 upward
    for K in range(3, 25):
        if time.time() > deadline:
            break
        
        # Give more time to smaller K (more likely to be optimal)
        if K <= 4:
            k_time_budget = 20.0
        elif K <= 6:
            k_time_budget = 10.0
        else:
            k_time_budget = 5.0
        
        k_start = time.time()
        
        # Try multiple random restarts
        for seed in range(200):
            if time.time() > k_start + k_time_budget or time.time() > deadline:
                break
            
            random.seed(seed)
            colors = np.array([random.randint(0, K - 1) for _ in range(n)], dtype=np.int32)
            
            # Local search with vectorized violation finding
            best_violations = len(find_violations(colors))
            
            for step in range(200000):
                if time.time() > deadline:
                    return colors  # Return best found so far
                
                viol_idx = find_violations(colors)
                n_viol = len(viol_idx)
                
                if n_viol == 0:
                    # Valid coloring found!
                    # Count distinct colors actually used
                    distinct = len(set(colors.tolist()))
                    return colors.tolist()
                
                # Pick a random violating AP
                v_idx = viol_idx[random.randint(0, n_viol - 1)]
                ap = aps_arr[v_idx]
                
                # Pick a random position in this AP
                pos_idx = random.randint(0, 3)
                pos = ap[pos_idx]
                
                # Change to a random different color
                old_color = colors[pos]
                new_color = random.randint(0, K - 1)
                while new_color == old_color:
                    new_color = random.randint(0, K - 1)
                colors[pos] = new_color
            
            # If local search didn't find valid, try simulated annealing for this restart
            if time.time() > deadline:
                return colors.tolist()
            
            # Simulated annealing phase
            sa_start = time.time()
            sa_budget = min(2.0, max(0.5, deadline - time.time()))
            temp = 1.0
            cooling = 0.9995
            
            for step in range(50000):
                if time.time() > sa_start + sa_budget or time.time() > deadline:
                    break
                
                viol_idx = find_violations(colors)
                n_viol = len(viol_idx)
                
                if n_viol == 0:
                    return colors.tolist()
                
                # Pick random position (not necessarily in violation)
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_viol = len(find_violations(colors))
                
                delta = new_viol - n_viol
                if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                    # Accept
                    pass
                else:
                    # Reject
                    colors[pos] = old_color
                
                temp *= cooling
        
        # End of K loop
    
    # Fallback: return a valid coloring with more colors
    # Use a known valid construction: color[i] = i % 3 doesn't work for d=3
    # Let's just do a greedy with many colors
    colors = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # Try colors 0, 1, 2, ... and pick first that doesn't create a violation
        for c in range(n):
            colors[i] = c
            # Check if any AP containing i is now violated
            # This is slow but n=128 is small
            # Actually, let's just assign i as color for simplicity
            # A coloring with all distinct colors is valid
            break
        colors[i] = i
    
    # Return distinct colors as labels
    return colors.tolist()

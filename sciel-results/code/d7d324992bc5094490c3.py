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
    deadline = start_time + 50.0  # Safe margin for 60s limit
    
    def find_violations(colors_arr):
        """Return indices of violating APs using vectorized NumPy."""
        ap_colors = colors_arr[aps_arr]  # (num_aps, 4)
        
        c_min = ap_colors.min(axis=1)
        c_max = ap_colors.max(axis=1)
        
        all_same = (c_min == c_max)
        
        in_range = ((ap_colors == c_min[:, None]) | (ap_colors == c_max[:, None])).all(axis=1)
        two_distinct = in_range & ~all_same
        
        violations = all_same | two_distinct
        return np.where(violations)[0]

    def count_violations(colors_arr):
        """Return count of violating APs."""
        ap_colors = colors_arr[aps_arr]
        c_min = ap_colors.min(axis=1)
        c_max = ap_colors.max(axis=1)
        all_same = (c_min == c_max)
        in_range = ((ap_colors == c_min[:, None]) | (ap_colors == c_max[:, None])).all(axis=1)
        two_distinct = in_range & ~all_same
        return np.sum(all_same | two_distinct)

    best_valid_colors = None
    best_valid_count = n  # Max possible

    # Try K from 3 upward
    for K in range(3, 10):
        if time.time() > deadline:
            break
        
        # Allocate time for this K
        # Smaller K is harder but more valuable. Give it more time.
        time_budget_for_k = 15.0 if K == 3 else 10.0 if K == 4 else 5.0
        k_start = time.time()
        
        for seed in range(1000):
            if time.time() > k_start + time_budget_for_k or time.time() > deadline:
                break
            
            random.seed(seed)
            colors = np.array([random.randint(0, K - 1) for _ in range(n)], dtype=np.int32)
            
            current_viol = count_violations(colors)
            
            # Simulated Annealing
            temp = 1.0
            cooling = 0.999
            steps = 2000
            
            for step in range(steps):
                if time.time() > deadline:
                    # If we have a valid solution, return it
                    if best_valid_colors is not None:
                        return best_valid_colors.tolist()
                    return colors.tolist()
                
                if current_viol == 0:
                    # Valid!
                    distinct = len(set(colors.tolist()))
                    if distinct < best_valid_count:
                        best_valid_count = distinct
                        best_valid_colors = colors.copy()
                    # Try to merge colors to reduce distinct count
                    # Simple merge: try to map one color to another
                    # This is complex, so we just record it and continue search for better K or same K
                    break
                
                # Pick random position
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                new_viol = count_violations(colors)
                
                delta = new_viol - current_viol
                if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                    current_viol = new_viol
                else:
                    colors[pos] = old_color
                
                temp *= cooling
            
            # If we found a valid solution in this restart, we can break early if K is small
            if current_viol == 0:
                distinct = len(set(colors.tolist()))
                if distinct < best_valid_count:
                    best_valid_count = distinct
                    best_valid_colors = colors.copy()
                
                # If we found a solution with K colors, and K is minimal (3), we can try to optimize it
                # But for now, just keep searching for potentially better (fewer colors) or same
                # If K==3 and we found valid, we can stop K loop if we want, but maybe 3 is not possible?
                # The task says "Minimize the number of labels".
                # If K=3 works, the min is 3 (since non-constant AP needs >=3 colors, so min is 3).
                # So if we find a valid K=3 coloring, we are done (score depends on raw value, which is likely number of colors or similar).
                # Wait, "Minimize the number of labels actually used".
                # If K=3 is valid, min labels is 3.
                # If K=3 is not valid, min labels is >=4.
                # So if we find a valid K=3, we can return immediately?
                # Yes, because 3 is the theoretical minimum for "at least 3 distinct colors".
                if K == 3:
                    return colors.tolist()

    # If no valid coloring found with small K, return the best valid one found
    if best_valid_colors is not None:
        return best_valid_colors.tolist()
    
    # Fallback: return a valid coloring with many colors (e.g., all distinct)
    # This ensures we don't return an invalid result
    colors = np.arange(n, dtype=np.int32)
    return colors.tolist()

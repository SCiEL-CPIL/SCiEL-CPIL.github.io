import numpy as np
import time
import random

def construct():
    n = 128
    deadline = time.time() + 55.0  # Safety margin
    
    # Precompute all non-constant 4-APs
    ap_indices = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            ap_indices.append([a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1])
            
    if not ap_indices:
        return [0] * n
        
    ap_matrix = np.array(ap_indices, dtype=np.int32)
    num_aps = ap_matrix.shape[0]
    
    # Vectorized cost function: counts number of bad APs (those with < 3 distinct colors)
    def compute_bad_mask_vectorized(colors):
        C = colors[ap_matrix] # shape (num_aps, 4)
        c0, c1, c2, c3 = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
        
        # 1 distinct color
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # 2 distinct colors:
        # Case 1: c0 == c1 and c2 == c3 and c0 != c2
        case1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
        # Case 2: c0 == c2 and c1 == c3 and c0 != c1
        case2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
        # Case 3: c0 == c3 and c1 == c2 and c0 != c1
        case3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
        # Case 4: Three of one, one of another
        # c0==c1==c2 != c3
        case4a = (c0 == c1) & (c1 == c2) & (c2 != c3)
        # c0==c1 != c2==c3 (covered by case1)
        # c0 != c1==c2==c3
        case4b = (c0 != c1) & (c1 == c2) & (c2 == c3)
        # c0==c1==c3 != c2
        case4c = (c0 == c1) & (c1 == c3) & (c2 != c1)
        # c0==c2==c3 != c1
        case4d = (c0 == c2) & (c2 == c3) & (c1 != c0)
        
        bad = all_same | case1 | case2 | case3 | case4a | case4b | case4c | case4d
        return bad

    # Try small number of colors
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
            
        # Run Simulated Annealing
        start_k_time = time.time()
        max_time_per_k = 15.0
        
        # Initial random coloring
        colors = np.random.randint(0, k, size=n, dtype=np.int32)
        bad_mask = compute_bad_mask_vectorized(colors)
        bad_count = np.sum(bad_mask)
        
        if bad_count == 0:
            return colors.tolist()
            
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.001
        
        # Precompute neighbor lists for faster updates if needed, but vectorized re-eval is fast enough for n=128
        # Number of APs is ~1000. Re-evaluating 1000 APs is very fast.
        
        while time.time() < start_k_time + max_time_per_k and time.time() < deadline and T > min_T:
            # Pick a random vertex to change
            v = random.randint(0, n - 1)
            old_color = colors[v]
            new_color = random.randint(0, k - 1)
            
            if new_color == old_color:
                continue
                
            # Apply change
            colors[v] = new_color
            
            # Calculate new cost
            new_bad_mask = compute_bad_mask_vectorized(colors)
            new_bad_count = np.sum(new_bad_mask)
            
            delta = new_bad_count - bad_count
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                bad_mask = new_bad_mask
                bad_count = new_bad_count
                if bad_count == 0:
                    return colors.tolist()
            else:
                # Revert
                colors[v] = old_color
                
            T *= cooling_rate
            
        # If SA didn't find a solution, try a few more restarts within the time limit for this k
        restarts = 0
        while time.time() < start_k_time + max_time_per_k and time.time() < deadline:
             # Random restart
             colors = np.random.randint(0, k, size=n, dtype=np.int32)
             bad_mask = compute_bad_mask_vectorized(colors)
             bad_count = np.sum(bad_mask)
             
             if bad_count == 0:
                 return colors.tolist()
                 
             # Run a short SA again
             T = 1.0
             steps = 1000
             for _ in range(steps):
                 if time.time() > deadline:
                     break
                 v = random.randint(0, n - 1)
                 old_color = colors[v]
                 new_color = random.randint(0, k - 1)
                 
                 if new_color == old_color:
                     continue
                     
                 colors[v] = new_color
                 new_bad_mask = compute_bad_mask_vectorized(colors)
                 new_bad_count = np.sum(new_bad_mask)
                 
                 delta = new_bad_count - bad_count
                 
                 if delta < 0 or random.random() < np.exp(-delta / T):
                     bad_mask = new_bad_mask
                     bad_count = new_bad_count
                     if bad_count == 0:
                         return colors.tolist()
                 else:
                     colors[v] = old_color
                     
                 T *= 0.99
                 
             restarts += 1
             if restarts > 20:
                 break

    # Fallback: Return a valid coloring if possible, otherwise best effort
    # A simple valid coloring for n=128 with 4 colors:
    # Use a known pattern or just return a random one if we fail, but we want to minimize colors.
    # If we couldn't find 3 or 4, try 5.
    for k in [5, 6]:
         colors = np.random.randint(0, k, size=n, dtype=np.int32)
         # Quick check
         if not np.any(compute_bad_mask_vectorized(colors)):
             return colors.tolist()
             
    # Absolute fallback: unique colors for each position (always valid, but max colors)
    return list(range(n))

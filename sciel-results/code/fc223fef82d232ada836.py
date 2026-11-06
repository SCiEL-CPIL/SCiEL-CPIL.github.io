import time
import random
import sys

def construct():
    N = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # 1. Precompute all Arithmetic Progressions
    # APs are stored as lists of 0-based indices
    aps = []
    # Map from index i to list of AP indices containing i
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            # Generate AP starting at a with step d
            # Indices are 0-based: a-1, a-1+d, ...
            start_idx = a - 1
            if start_idx >= N:
                continue
            ap_indices = []
            curr = start_idx
            while curr < N:
                ap_indices.append(curr)
                curr += d
            if len(ap_indices) > 0:
                ap_id = len(aps)
                aps.append(ap_indices)
                for idx in ap_indices:
                    aps_containing[idx].append(ap_id)
                    
    num_aps = len(aps)
    
    # 2. Initialization
    # Use a simple random initialization
    rng = random.Random(42)
    s = [rng.choice((-1, 1)) for _ in range(N)]
    
    # Calculate initial sums
    ap_sums = [0] * num_aps
    for ap_id, ap_indices in enumerate(aps):
        total = 0
        for idx in ap_indices:
            total += s[idx]
        ap_sums[ap_id] = total
        
    # Histogram of absolute sums
    # Max possible sum is N=256
    max_val = N
    hist = [0] * (max_val + 1)
    for total in ap_sums:
        hist[abs(total)] += 1
        
    # Find current max
    current_max = max_val
    while current_max > 0 and hist[current_max] == 0:
        current_max -= 1
        
    best_max = current_max
    best_s = s[:]
    
    # 3. Local Search (Simulated Annealing / Hill Climbing)
    # We will try to minimize the max absolute sum.
    # Temperature for SA
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        i = rng.randrange(N)
        
        # Calculate change in sums for all APs containing i
        # Flipping s[i] changes sum of AP by -2 * s[i] (old value)
        # Let old_val = s[i]
        # new_val = -old_val
        # delta_sum = -2 * old_val
        
        old_val = s[i]
        delta = -2 * old_val
        
        # Update sums and histogram
        # We need to check if the new max is lower
        # We must be careful with the histogram updates to be reversible or check validity
        
        # Let's simulate the flip
        # List of (ap_id, old_sum, new_sum)
        changes = []
        # We need to update hist for old sums
        # And check if current_max decreases
        
        # Optimization: Only accept if it doesn't increase the max, 
        # or with probability exp(-delta_max/T) if it increases.
        
        # To evaluate the new max, we need to know the new histogram.
        # Since degree is small (~200), we can apply changes, check max, and revert if needed.
        
        # Apply changes
        new_max = current_max
        
        # Track hist changes to revert easily
        # hist_diff = {}
        
        # We'll just modify hist and ap_sums, and keep track to revert.
        # Actually, it's safer to compute the new max without modifying global state,
        # but that's hard. Modifying and reverting is standard.
        
        revert_list = []
        
        for ap_id in aps_containing[i]:
            old_sum = ap_sums[ap_id]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            # Update hist
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            # Track for revert
            revert_list.append((ap_id, old_sum, old_abs, new_abs))
            
            # Update ap_sums
            ap_sums[ap_id] = new_sum
            
        # Determine new_max
        # If current_max is affected, we might need to adjust
        # If hist[current_max] becomes 0, we need to find next lower non-zero.
        # If we added to a value > current_max, new_max increases.
        
        # Check if max increased
        # The max can only increase if we added to a bucket > current_max.
        # Or if we removed from current_max and it became 0, max decreases.
        
        # Efficient max update:
        # 1. If any new_abs > current_max, new_max = max(new_max, new_abs)
        # 2. If hist[current_max] == 0, decrement new_max until non-zero.
        
        # Let's just scan down from current_max if needed, and check up if needed.
        # Since changes are small, we can just check the affected buckets.
        
        # Actually, simplest robust way:
        # new_max = current_max
        # if hist[new_max] == 0:
        #     while new_max > 0 and hist[new_max] == 0:
        #         new_max -= 1
        # # Check if any new_abs is larger than new_max
        # for _, _, _, new_abs in revert_list:
        #     if new_abs > new_max:
        #         new_max = new_abs
        #         break # One is enough to know it's higher? No, need max.
        
        # Better:
        max_new_abs = 0
        for _, _, _, new_abs in revert_list:
            if new_abs > max_new_abs:
                max_new_abs = new_abs
                
        if max_new_abs > new_max:
            new_max = max_new_abs
        else:
            # Check if current_max bucket is empty
            if hist[new_max] == 0:
                while new_max > 0 and hist[new_max] == 0:
                    new_max -= 1
        
        # Decision
        if new_max < current_max:
            # Always accept improvement
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_s = s[:]
            # Update s
            s[i] = -old_val
            # Don't revert ap_sums/hist
        else:
            # Simulated Annealing
            delta_max = new_max - current_max
            if delta_max == 0:
                # Accept with some probability to escape local minima?
                # Or just accept to explore?
                # Let's accept 50% of the time if equal, to shuffle.
                if rng.random() < 0.5:
                    s[i] = -old_val
                    current_max = new_max
                else:
                    # Revert
                    for ap_id, old_sum, old_abs, new_abs in revert_list:
                        ap_sums[ap_id] = old_sum
                        hist[old_abs] += 1
                        hist[new_abs] -= 1
            else:
                # delta_max > 0
                # Accept with prob exp(-delta_max / T)
                # Since delta_max is integer >= 1, and T is small, this is rare.
                if rng.random() < pow(2.71828, -delta_max / T):
                    s[i] = -old_val
                    current_max = new_max
                else:
                    # Revert
                    for ap_id, old_sum, old_abs, new_abs in revert_list:
                        ap_sums[ap_id] = old_sum
                        hist[old_abs] += 1
                        hist[new_abs] -= 1
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min

    return best_s

# EVOLVE-BLOCK-START
import time
import random

def construct():
    N = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # 1. Precompute all Arithmetic Progressions
    aps = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            start_idx = a - 1
            if start_idx >= N:
                continue
            ap_indices = []
            curr = start_idx
            while curr < N:
                ap_indices.append(curr)
                curr += d
            if len(ap_indices) > 0:
                ap_id = len(aps)
                aps.append(ap_indices)
                for idx in ap_indices:
                    aps_containing[idx].append(ap_id)
                    
    num_aps = len(aps)
    
    # 2. Initialization
    rng = random.Random(42)
    s = [rng.choice((-1, 1)) for _ in range(N)]
    
    ap_sums = [0] * num_aps
    for ap_id, ap_indices in enumerate(aps):
        total = 0
        for idx in ap_indices:
            total += s[idx]
        ap_sums[ap_id] = total
        
    max_val = N
    hist = [0] * (max_val + 1)
    for total in ap_sums:
        hist[abs(total)] += 1
        
    current_max = max_val
    while current_max > 0 and hist[current_max] == 0:
        current_max -= 1
        
    best_max = current_max
    best_s = s[:]
    
    # 3. Local Search
    T = 2.0
    T_min = 0.01
    alpha = 0.9995
    
    while time.time() - start_time < time_limit:
        i = rng.randrange(N)
        
        old_val = s[i]
        delta = -2 * old_val
        
        revert_list = []
        
        for ap_id in aps_containing[i]:
            old_sum = ap_sums[ap_id]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            revert_list.append((ap_id, old_sum, old_abs, new_abs))
            ap_sums[ap_id] = new_sum
        
        new_max = current_max
        
        max_new_abs = 0
        for _, _, _, new_abs in revert_list:
            if new_abs > max_new_abs:
                max_new_abs = new_abs
                
        if max_new_abs > new_max:
            new_max = max_new_abs
        else:
            if hist[new_max] == 0:
                while new_max > 0 and hist[new_max] == 0:
                    new_max -= 1
        
        if new_max < current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_s = s[:]
            s[i] = -old_val
        else:
            delta_max = new_max - current_max
            if delta_max == 0:
                if rng.random() < 0.5:
                    s[i] = -old_val
                    current_max = new_max
                else:
                    for ap_id, old_sum, old_abs, new_abs in revert_list:
                        ap_sums[ap_id] = old_sum
                        hist[old_abs] += 1
                        hist[new_abs] -= 1
            else:
                if rng.random() < pow(2.71828, -delta_max / T):
                    s[i] = -old_val
                    current_max = new_max
                else:
                    for ap_id, old_sum, old_abs, new_abs in revert_list:
                        ap_sums[ap_id] = old_sum
                        hist[old_abs] += 1
                        hist[new_abs] -= 1
        
        T *= alpha
        if T < T_min:
            T = T_min

    return best_s
# EVOLVE-BLOCK-END

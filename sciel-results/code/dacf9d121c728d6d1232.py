import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin

    # 1. Precompute APs
    aps = []
    aps_containing = [[] for _ in range(n)]
    
    # Generate all APs of length >= 2
    # Note: Singletons are trivial (sum=1). We can ignore them or include them.
    # Including them doesn't change the max if max > 1.
    # Let's generate all APs with length >= 2 to save memory/time, 
    # but the problem says "including one". 
    # Since |1| = 1, if we find a solution with max >= 1, singletons don't matter.
    # Any non-trivial AP with length 2 has sum in {-2, 0, 2}. 
    # So the max will be at least 1.
    
    for d in range(1, n):
        for s in range(n):
            curr = s
            prefix = []
            while curr < n:
                prefix.append(curr)
                curr += d
            if len(prefix) >= 2:
                ap_id = len(aps)
                aps.append(prefix)
                for idx in prefix:
                    aps_containing[idx].append(ap_id)
                
    # Also add singletons if we want to be strictly correct, 
    # but they are dominated by any length-2 AP if the discrepancy is > 1.
    # However, if we somehow get a discrepancy of 0 (impossible for sum of signs? 
    # Sum of even number of signs can be 0. Sum of odd number cannot be 0.
    # Max abs sum for length 2 is 2.
    # So max will be >= 1.
    # Let's just include length >= 2.
    
    num_aps = len(aps)
    
    # 2. Initialize
    rng = random.Random(42)
    x = [rng.choice([-1, 1]) for _ in range(n)]
    
    # Initial sums
    sums = np.zeros(num_aps, dtype=np.int32)
    for k, ap in enumerate(aps):
        s = 0
        for idx in ap:
            s += x[idx]
        sums[k] = s
        
    current_max = int(np.max(np.abs(sums)))
    
    best_x = x[:]
    best_sums = sums.copy()
    best_max = current_max
    
    # 3. Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute numpy arrays for aps_containing for faster indexing?
    # aps_containing[i] is a list of ints. 
    # We can convert to numpy arrays once.
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            # Restart with new T if we've cooled down too much? 
            # Or just break and do a new random restart if time permits.
            # For now, let's just continue with min_T or restart.
            # A better strategy: if T is low, we are stuck. 
            # Let's implement a "restart" logic if no improvement in last X seconds?
            # Simpler: just keep going.
            pass
            
        i = rng.randint(0, n - 1)
        old_val = x[i]
        new_val = -old_val
        delta = -2 * old_val  # Change in sum for APs containing i
        
        indices = aps_containing_np[i]
        if len(indices) == 0:
            continue
            
        sums[indices] += delta
        new_max = int(np.max(np.abs(sums)))
        delta_max = new_max - current_max
        
        if delta_max < 0:
            x[i] = new_val
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_x = x[:]
                best_sums = sums.copy()
        else:
            # Accept with probability exp(-delta_max / T)
            if rng.random() < np.exp(-delta_max / T):
                x[i] = new_val
                current_max = new_max
            else:
                # Revert
                sums[indices] -= delta
        
        T *= cooling

    return best_x

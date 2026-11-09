import numpy as np
import time
import random

# Precompute all arithmetic progressions in [1, 256]
N = 256
aps = []
index_to_aps = [[] for _ in range(N)]

# Generate APs
# An AP is (start, step, length)
# Indices are 0-based: start-1 to start-1 + (length-1)*step
for step in range(1, N + 1):
    for start in range(1, N + 1):
        # Max length such that start + (L-1)*step <= N
        max_len = (N - start) // step + 1
        for length in range(1, max_len + 1):
            indices = [start - 1 + j * step for j in range(length)]
            ap_id = len(aps)
            aps.append(indices)
            for idx in indices:
                index_to_aps[idx].append(ap_id)

num_aps = len(aps)
# Precompute the AP indices for each position as a numpy array for fast access
# ap_indices_for_pos[i] is an array of AP IDs that contain index i
ap_indices_for_pos = [np.array(index_to_aps[i], dtype=np.int32) for i in range(N)]

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    best_signs = None
    best_score = float('inf')
    
    # Run multiple restarts
    seed = 42
    while time.time() - start_time < time_limit:
        # Check if we should restart
        # We do a local search for a fixed duration or until stuck
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        
        # Initialize sums
        # sums[k] is the sum of signs over the k-th AP
        sums = np.zeros(num_aps, dtype=np.int16)
        
        # Efficiently initialize sums
        # For each AP, sum the signs
        # This is O(num_aps * avg_len). 
        # Alternatively, accumulate by index.
        # Let's use the index_to_aps mapping.
        # For each index i, signs[i] is added to all APs in index_to_aps[i]
        for i in range(N):
            if signs[i] != 0:
                sums[ap_indices_for_pos[i]] += signs[i]
                
        current_max = np.max(np.abs(sums))
        
        # Local Search
        last_improvement_time = time.time()
        no_improve_count = 0
        max_no_improve = 5000 # Restart if stuck for this many iterations
        
        while time.time() - start_time < time_limit:
            # Pick a random index to flip
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            signs[i] = new_sign
            
            # Update sums
            delta = new_sign - old_sign # +2 or -2
            sums[ap_indices_for_pos[i]] += delta
            
            # Calculate new max
            new_max = np.max(np.abs(sums))
            
            if new_max < current_max:
                current_max = new_max
                last_improvement_time = time.time()
                no_improve_count = 0
                if current_max < best_score:
                    best_score = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_sign
                sums[ap_indices_for_pos[i]] -= delta
                no_improve_count += 1
                
            # Restart condition: if stuck for too long
            if no_improve_count > max_no_improve:
                break
                
            # If best_score is very low, we might want to break early?
            # Theoretical min is likely around 4-6. If we hit 4, we can stop or keep trying for 3?
            # 3 is extremely unlikely for N=256.
            if best_score <= 4:
                # We found a very good solution. Keep trying for a bit more?
                # Or break to save time?
                # Let's break if we hit 4, as 3 is very hard.
                break

        seed += 1
        if best_score <= 4:
            break

    # Ensure we return a valid list of ints
    if best_signs is None:
        # Fallback to random if something went wrong (shouldn't happen)
        return [random.choice([-1, 1]) for _ in range(N)]
        
    return [int(x) for x in best_signs]

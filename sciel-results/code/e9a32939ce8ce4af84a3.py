import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)

    # Convert to numpy array for vectorized operations
    ap_array = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    # For each bit position, store the indices of APs it's in, and for each such AP,
    # store the other 3 positions as a numpy array for fast checking
    bit_ap_info = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for j in range(4):
            pos = ap[j]
            others = np.array([ap[k] for k in range(4) if k != j], dtype=np.int32)
            bit_ap_info[pos].append((idx, others))

    # Function to count monochromatic APs (vectorized)
    def count_mono(x):
        x = np.asarray(x, dtype=np.int8)
        vals = x[ap_array]  # (num_aps, 4)
        return int(np.sum(np.all(vals == vals[:, 0:1], axis=1)))

    # Fast delta for flipping bit i using numpy
    def delta_flip(x, i):
        old_val = x[i]
        new_val = 1 - old_val
        delta = 0
        for (ap_idx, others) in bit_ap_info[i]:
            other_vals = x[others]
            was_mono = np.all(other_vals == old_val)
            is_mono = np.all(other_vals == new_val)
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    # Compute delta for flipping a set of bits simultaneously
    def delta_flip_set(x, bits):
        """Compute cost change when flipping all bits in the set simultaneously."""
        old_vals = {b: x[b] for b in bits}
        new_vals = {b: 1 - x[b] for b in bits}
        
        # Find all APs affected by any of these bits
        affected_aps = set()
        for b in bits:
            for (ap_idx, others) in bit_ap_info[b]:
                affected_aps.add(ap_idx)
        
        delta = 0
        for ap_idx in affected_aps:
            ap = aps[ap_idx]
            # Check if this AP is monochromatic before
            was_mono = True
            for pos in ap:
                if pos in old_vals:
                    if x[pos] != old_vals[pos]:  # should be same
                        pass
                else:
                    # Use current value
                    pass
            # Simpler: just check all 4 positions with their "before" values
            before_vals = []
            after_vals = []
            for pos in ap:
                if pos in old_vals:
                    before_vals.append(old_vals[pos])
                    after_vals.append(new_vals[pos])
                else:
                    before_vals.append(x[pos])
                    after_vals.append(x[pos])
            
            was_mono = len(set(before_vals)) == 1
            is_mono = len(set(after_vals)) == 1
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        
        return delta

    # Simulated Annealing with proper 1-flip and 2-flip moves
    def simulated_annealing(initial_x, time_limit, T_start=8.0, T_end=0.1):
        x = np.array(initial_x, dtype=np.int8)
        current_cost = count_mono(x)
        best_x = x.copy()
        best_cost = current_cost
        
        T = T_start
        step = 0
        total_steps_estimate = 500000
        cool_rate = (T_end / T_start) ** (1.0 / total_steps_estimate)
        
        # Pre-generate random numbers in batches for speed
        batch_size = 10000
        rng = np.random.default_rng()
        
        while time.time() < time_limit:
            # Generate a batch of random choices
            rand_batch = rng.random(batch_size)
            idx_batch = rng.integers(0, N, size=batch_size)
            
            for batch_idx in range(batch_size):
                if time.time() >= time_limit:
                    break
                    
                rand_val = rand_batch[batch_idx]
                
                if rand_val < 0.25:
                    # 2-flip move
                    i1 = idx_batch[batch_idx]
                    i2 = idx_batch[(batch_idx + 1) % batch_size]
                    if i1 == i2:
                        # Generate a new i2
                        i2 = rng.integers(0, N)
                        if i1 == i2:
                            continue
                    
                    d = delta_flip_set(x, (i1, i2))
                    
                    if d < 0 or rng.random() < np.exp(-d / T):
                        x[i1] = 1 - x[i1]
                        x[i2] = 1 - x[i2]
                        current_cost += d
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_x = x.copy()
                else:
                    # 1-flip move
                    i = idx_batch[batch_idx]
                    d = delta_flip(x, i)
                    
                    if d < 0 or rng.random() < np.exp(-d / T):
                        x[i] = 1 - x[i]
                        current_cost += d
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_x = x.copy()
                
                step += 1
                T *= cool_rate
                if T < T_end:
                    T = T_end
        
        return best_x, best_cost

    # Initialize with diverse starting points
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    tm_x = np.array(thue_morse(N), dtype=np.int8)
    rand_x = np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)
    # Periodic patterns with different densities
    per3_x = np.array([i % 3 < 1 for i in range(N)], dtype=np.int8)  # 1/3 density
    per4_x = np.array([i % 4 < 2 for i in range(N)], dtype=np.int8)  # 1/2 density blocks
    per5_x = np.array([i % 5 < 2 for i in range(N)], dtype=np.int8)  # 2/5 density
    
    # Also try a "block" pattern: alternate blocks of 0s and 1s
    block_x = np.zeros(N, dtype=np.int8)
    block_size = 8
    for b in range(N // block_size):
        if b % 2 == 1:
            block_x[b*block_size:(b+1)*block_size] = 1
    
    best_overall_x = None
    best_overall_cost = float('inf')
    
    # Run SA on multiple starts, splitting time
    starts = [rand_x, tm_x, per3_x, per4_x, per5_x, block_x]
    n_starts = len(starts)
    time_per_start = (deadline - start_time) / n_starts
    
    for s_idx, init_x in enumerate(starts):
        t_limit = start_time + (s_idx + 1) * time_per_start
        if time.time() < t_limit:
            x_best, c_best = simulated_annealing(init_x, t_limit)
            if c_best < best_overall_cost:
                best_overall_cost = c_best
                best_overall_x = x_best
    
    # Fallback
    if best_overall_x is None:
        best_overall_x = rand_x
    
    # Final polish: multi-pass greedy local search
    x = np.array(best_overall_x, dtype=np.int8)
    improved = True
    passes = 0
    while improved and time.time() < deadline - 3 and passes < 20:
        improved = False
        passes += 1
        for i in range(N):
            d = delta_flip(x, i)
            if d < 0:
                x[i] = 1 - x[i]
                improved = True
        # Also try 2-flip improvements
        for i in range(N):
            for j in range(i+1, N):
                d = delta_flip_set(x, (i, j))
                if d < 0:
                    x[i] = 1 - x[i]
                    x[j] = 1 - x[j]
                    improved = True
    
    # Check if polished solution is better
    final_cost = count_mono(x)
    if final_cost < best_overall_cost:
        best_overall_x = x
        best_overall_cost = final_cost
    
    # Return as list of ints
    return [int(v) for v in best_overall_x]

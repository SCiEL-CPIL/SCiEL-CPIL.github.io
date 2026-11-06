import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    N = 256
    
    # 1. Precompute ALL APs and inverse mapping
    ap_indices = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            # Every prefix is an AP
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # Convert to numpy arrays for faster access
    ap_indices_np = [np.array(lst, dtype=np.int32) for lst in ap_indices]
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # 2. Continuous Optimization (Gradient Descent on Sum of Squares)
    def continuous_optimize(x_init, time_budget):
        x = x_init.astype(np.float64).copy()
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(x)
        
        # Precompute sums
        sums = np.zeros(num_aps, dtype=np.float64)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(x[idx_list])
            
        # Loss is sum(sums**2)
        
        while time.time() - start_time < time_budget:
            # Compute Gradient
            # grad_i = 2 * sum(sums[k] for k in aps_containing[i])
            grad = np.zeros(N, dtype=np.float64)
            for i in range(N):
                # Sum of sums of APs containing i
                # This is O(num_aps_per_i)
                # We can vectorize this slightly by using the precomputed lists
                # But python loop over N=256 is fast enough if inner is numpy sum
                if len(aps_containing_np[i]) > 0:
                    grad[i] = 2.0 * np.sum(sums[aps_containing_np[i]])
                else:
                    grad[i] = 0.0
            
            # Update
            v = momentum * v - lr * grad
            x += v
            
            # Optional: Project or constrain? 
            # The objective is convex in x for fixed APs? No, sum of squares of linear forms is convex.
            # So GD should work well.
            
            # Update sums incrementally?
            # Sums change as x changes. 
            # sums_new = sums_old + delta_sums
            # delta_sums[r] = sum(v[idx] for idx in ap_indices[r])
            # This is also O(total_ap_length).
            # It's simpler to recompute sums from scratch every few steps or every step if fast enough.
            # Total length of all APs is approx N^2/2 * avg_len? 
            # Actually, sum of lengths of all APs is approx N^2 * log(N) / 2?
            # For N=256, it's manageable.
            
            # Recompute sums
            for r, idx_list in enumerate(ap_indices_np):
                sums[r] = np.sum(x[idx_list])
                
            # Adaptive learning rate
            # Check loss
            loss = np.sum(sums**2)
            # Simple decay
            lr *= 0.99
            
        return x

    # 3. Discrete Optimization (Local Search)
    def discrete_optimize(s_init, time_budget):
        s = s_init.copy()
        
        # Compute initial sums
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
            
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        # Histogram for max tracking
        max_possible_sum = N
        hist = np.zeros(max_possible_sum + 2, dtype=np.int32)
        for val in np.abs(sums):
            hist[val] += 1
            
        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
            
        rng = random.Random(42)
        indices_list = list(range(N))
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            improved = False
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                    
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                if len(affected) == 0:
                    continue
                    
                new_sums_affected = sums[affected] + delta
                new_abs_affected = np.abs(new_sums_affected)
                
                # Greedy: Only accept if it reduces the max
                # Or if it doesn't increase the max and reduces the sum of squares (optional)
                old_max = current_max
                # Check if new max is lower
                # We need to know if any AP goes above current_max
                # And if the global max drops.
                
                # Quick check: if any new_abs > current_max, skip
                if np.any(new_abs_affected > current_max):
                    continue
                
                # Apply tentatively
                old_abs_affected = np.abs(sums[affected])
                
                # Update hist
                for j, ap_idx in enumerate(affected):
                    old_abs = old_abs_affected[j]
                    new_abs = new_abs_affected[j]
                    if old_abs != new_abs:
                        hist[old_abs] -= 1
                        hist[new_abs] += 1
                        
                sums[affected] = new_sums_affected
                s[i] = new_val
                
                new_global_max = get_max_from_hist()
                
                if new_global_max < current_max:
                    current_max = new_global_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                    improved = True
                else:
                    # Revert if no improvement in max
                    # Revert hist
                    for j, ap_idx in enumerate(affected):
                        old_abs = old_abs_affected[j]
                        new_abs = new_abs_affected[j]
                        if old_abs != new_abs:
                            hist[new_abs] -= 1
                            hist[old_abs] += 1
                    # Revert sums and s
                    sums[affected] -= delta
                    s[i] = old_val
                    
            if not improved:
                # Stagnation: Try random flips or break
                # For simplicity, if no improvement in a full pass, we might be stuck.
                # We can try a few random flips to escape.
                if rng.random() < 0.1:
                    # Flip a random position regardless of max increase, then try to recover?
                    # Or just break and rely on other restarts.
                    pass

        return best_s, best_score

    # 4. Main Logic
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N
    
    # Initialize candidates
    candidates = []
    
    # Candidate 1: Thue-Morse
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq
    tm = thue_morse(N)
    candidates.append(tm.astype(np.float64))
    
    # Candidate 2: Random
    candidates.append(np.array([rng.choice([-1.0, 1.0]) for _ in range(N)], dtype=np.float64))
    
    # Candidate 3: Random Normal (scaled)
    candidates.append(np.array([rng.gauss(0, 1) for _ in range(N)], dtype=np.float64))
    
    # Process candidates
    for x_init in candidates:
        time_left = time_limit - (time.time() - start_time)
        if time_left < 2.0:
            break
            
        # Split time: 50% continuous, 50% discrete polish
        t_cont = time_left * 0.5
        t_disc = time_left * 0.5
        
        x_opt = continuous_optimize(x_init, t_cont)
        
        # Discretize
        s_disc = np.where(x_opt > 0, 1, -1).astype(np.int8)
        
        # Polish
        s_final, score_final = discrete_optimize(s_disc, t_disc)
        
        if score_final < best_score_overall:
            best_score_overall = score_final
            best_s_overall = s_final
            
    # If no candidates processed or time ran out, use Thue-Morse as fallback
    if best_s_overall is None:
        return [int(x) for x in tm]
        
    return [int(x) for x in best_s_overall]

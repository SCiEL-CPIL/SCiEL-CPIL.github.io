import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 90s limit

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
            
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # Convert to numpy arrays for faster indexing
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    ap_indices_np = [np.array(lst, dtype=np.int32) for lst in ap_indices]

    # 2. Initialization: Thue-Morse sequence
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    # 3. Core Optimization with dual-target + annealing
    def optimize(s_init, time_budget, rng):
        s = s_init.copy()
        
        # Compute initial sums for all APs
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
            
        abs_sums = np.abs(sums)
        current_max = int(np.max(abs_sums))
        
        # Maintain histogram of absolute sum values
        counts = np.bincount(abs_sums, minlength=N + 2)
        
        best_s = s.copy()
        best_score = current_max
        
        indices_list = list(range(N))
        
        # Dual-target system
        target_D = max(0, current_max - 1)
        
        # Annealing parameters
        T = 1.0
        T_min = 0.001
        T_max = 5.0
        cooling = 0.995
        heating = 1.02
        stuck_counter = 0
        max_stuck = 500
        
        # Main optimization loop
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                
                # Save old sums for affected APs
                old_sums = sums[affected].copy()
                
                # Apply flip to sums (simulate)
                new_sums = sums[affected] + delta
                new_abs = np.abs(new_sums)
                
                # Hard ceiling: never allow exceeding current_max
                if np.any(new_abs > current_max):
                    continue
                
                # Check against soft target
                violates_target = np.any(new_abs > target_D)
                
                if not violates_target:
                    # Accept: satisfies the stricter target
                    # Update histogram
                    old_abs = np.abs(old_sums)
                    changed_mask = old_abs != new_abs
                    if np.any(changed_mask):
                        changed_old = old_abs[changed_mask]
                        changed_new = new_abs[changed_mask]
                        np.add.at(counts, changed_old, -1)
                        np.add.at(counts, changed_new, 1)
                    
                    # Commit
                    sums[affected] = new_sums
                    s[i] = new_val
                    stuck_counter = 0
                    
                    # Cool down when making progress
                    T = max(T_min, T * cooling)
                    
                    # Check if global max decreased
                    if counts[current_max] == 0:
                        new_max = current_max
                        while new_max > 0 and counts[new_max] == 0:
                            new_max -= 1
                        current_max = new_max
                        target_D = max(0, current_max - 1)
                        
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                        
                        # Boost temperature slightly after finding improvement
                        T = min(T_max, T * 1.5)
                
                else:
                    # Move violates target_D but is within current_max
                    # Accept with temperature-dependent probability
                    # Compute how much worse this move is
                    max_violation = int(np.max(new_abs)) - target_D
                    # Probability of accepting: higher T -> higher acceptance
                    p_accept = min(1.0, T / (1.0 + max_violation * 0.5))
                    
                    if rng.random() < p_accept:
                        # Update histogram
                        old_abs = np.abs(old_sums)
                        changed_mask = old_abs != new_abs
                        if np.any(changed_mask):
                            changed_old = old_abs[changed_mask]
                            changed_new = new_abs[changed_mask]
                            np.add.at(counts, changed_old, -1)
                            np.add.at(counts, changed_new, 1)
                        
                        # Commit
                        sums[affected] = new_sums
                        s[i] = new_val
                        stuck_counter += 1
                        
                        # Heat up when stuck
                        T = min(T_max, T * heating)
                    else:
                        stuck_counter += 1
                        T = min(T_max, T * heating)
                    
                    # Periodic check if we can lower the target
                    if stuck_counter % 100 == 0:
                        # Check if all APs with |sum| == target_D + 1 are now <= target_D
                        # i.e., can we lower target_D?
                        if counts[target_D + 1] == 0 and target_D > 0:
                            target_D -= 1
                            stuck_counter = 0
                            T = max(T_min, T * cooling)
        
        return best_s, best_score

    rng = random.Random(42)
    
    # Run 1: Start from Thue-Morse
    tm = thue_morse(N)
    s1, score1 = optimize(tm, time_limit * 0.35, rng)
    best_s = s1
    best_score = score1

    # Run 2: Start from random
    if time.time() - start_time < time_limit * 0.65:
        rand_init = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = optimize(rand_init, time_limit * 0.25, rng)
        if score2 < best_score:
            best_score = score2
            best_s = s2

    # Run 3: Refine the best solution found
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        s3, score3 = optimize(best_s, remaining - 0.5, rng)
        if score3 < best_score:
            best_score = score3
            best_s = s3

    return [int(x) for x in best_s]

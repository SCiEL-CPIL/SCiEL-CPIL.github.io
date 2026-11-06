import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    N = 256

    # 1. Precompute ALL APs and inverse mapping
    # AP defined by start s (1-indexed), step d.
    # Indices in 0-indexed: s-1, s-1+d, ...
    
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
    
    # Precompute lengths of APs for potential weighting or normalization
    ap_lengths = np.array([len(lst) for lst in ap_indices_np], dtype=np.int32)

    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    def compute_smooth_loss(sums, beta):
        # sums is an array of AP sums
        # Loss = beta * log(sum(exp(|sum|/beta)))
        # To avoid overflow, use log-sum-exp trick
        abs_sums = np.abs(sums)
        if beta == 0:
            return np.max(abs_sums)
        
        max_abs = np.max(abs_sums)
        if max_abs == 0:
            return 0.0
            
        # L = beta * (max_abs/beta + log(sum(exp((|sum|-max_abs)/beta))))
        # L = max_abs + beta * log(sum(exp((|sum|-max_abs)/beta)))
        
        exp_terms = np.exp((abs_sums - max_abs) / beta)
        log_sum_exp = np.log(np.sum(exp_terms))
        
        return max_abs + beta * log_sum_exp

    def optimize_smooth(s_init, time_budget, rng):
        s = s_init.copy()
        
        # Initial sums
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
            
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        # Smooth optimization parameters
        beta_start = 50.0
        beta_end = 1.0
        T_start = 10.0
        T_end = 0.1
        
        # We will anneal beta and T
        # Number of steps is determined by time budget
        # Estimate steps per second: ~1000-5000 depending on overhead
        # Let's aim for ~5000 steps per second.
        
        elapsed = time.time() - start_time
        remaining_time = time_budget - elapsed
        if remaining_time <= 0:
            return best_s, best_score
            
        # Total steps to attempt
        steps_per_sec = 3000
        total_steps = int(remaining_time * steps_per_sec)
        
        # Annealing schedule
        # Linear decay for simplicity, or exponential
        # Let's use exponential decay for T and beta
        
        indices_list = list(range(N))
        
        for step in range(total_steps):
            if time.time() - start_time > time_budget:
                break
                
            # Current beta and T
            progress = step / total_steps
            beta = beta_start * (beta_end / beta_start) ** progress
            T = T_start * (T_end / T_start) ** progress
            
            # Compute gradient-like signal
            # grad_i = sum_{k ni i} (dL/dS_k) * sign(S_k)
            # dL/dS_k = sign(S_k) * exp(|S_k|/beta) / sum(exp(|S_j|/beta))
            
            # We can compute the weights w_k = exp(|S_k|/beta)
            # Then normalize. But we don't need the exact gradient value, 
            # we need the change in loss for a flip.
            
            # Change in loss for flipping x_i:
            # For each k containing i, S_k -> S_k + delta_i
            # delta_i = -2 * s[i]
            
            # Instead of full gradient, let's use a heuristic:
            # Flip i if it reduces the sum of squares of large sums?
            # Or use the smooth loss change directly.
            
            # Efficient calculation of loss change for all i is hard.
            # Let's pick a random i and calculate the change for it.
            
            i = rng.choice(indices_list)
            old_val = s[i]
            delta = -2 * old_val
            new_val = -old_val
            
            affected = aps_containing_np[i]
            
            # Current loss contribution of affected APs?
            # Actually, we need the global loss change.
            # L = max_abs + beta * log(sum(exp((|S|-max_abs)/beta)))
            # This is complex to update incrementally for the log-sum-exp part.
            
            # Alternative: Use the "soft" sum of squares as a proxy for intermediate stages,
            # and switch to max-sum local search later.
            # Or, use the fact that for large beta, L approximates sum(S_k^2)/2beta.
            # Let's use sum of squares as the objective for the smooth phase.
            # Obj = sum(S_k^2)
            # Change in Obj for flipping i:
            # Sum_{k ni i} (S_k + delta)^2 - S_k^2
            # = Sum_{k ni i} (2*S_k*delta + delta^2)
            # = delta * (2 * Sum_{k ni i} S_k + delta * len(affected))
            
            # Let's compute this.
            affected_sums = sums[affected]
            delta_obj = delta * (2 * np.sum(affected_sums) + delta * len(affected))
            
            # If delta_obj < 0, accept.
            # If delta_obj > 0, accept with prob exp(-delta_obj / T)
            
            if delta_obj < 0:
                accept = True
            else:
                # T is in same units as delta_obj?
                # delta_obj can be large. Let's scale T.
                # Max delta_obj is roughly 2 * max_S * N * N.
                # Let's use a scaled temperature.
                scale_T = T * 100.0
                if scale_T == 0:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-delta_obj / scale_T)
            
            if accept:
                # Update sums
                sums[affected] += delta
                s[i] = new_val
                
                # Check for improvement in true max sum
                # Do this periodically to save time
                if step % 100 == 0:
                    current_max = int(np.max(np.abs(sums)))
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                        
        # Final local search on max sum
        # Use the best_s found so far
        s = best_s
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
            
        current_max = int(np.max(np.abs(sums)))
        best_score = current_max
        best_s = s.copy()
        
        # Strict local search
        indices_list = list(range(N))
        max_iter = 50000
        for _ in range(max_iter):
            if time.time() - start_time > time_budget:
                break
                
            i = rng.choice(indices_list)
            old_val = s[i]
            delta = -2 * old_val
            new_val = -old_val
            affected = aps_containing_np[i]
            
            new_sums_affected = sums[affected] + delta
            new_abs_affected = np.abs(new_sums_affected)
            
            # Check if max increases
            if np.any(new_abs_affected > current_max):
                continue
                
            # If max doesn't increase, accept if it decreases max or just accept to explore
            # Let's accept if it decreases the sum of squares of sums near the max
            # Or just accept any move that doesn't increase max (random walk on plateau)
            
            # To ensure progress, let's only accept if it reduces the count of APs with |sum| == current_max
            # Or if it reduces the sum of squares.
            
            old_sq_sum = np.sum(sums[affected]**2)
            new_sq_sum = np.sum(new_sums_affected**2)
            
            if new_sq_sum < old_sq_sum:
                sums[affected] = new_sums_affected
                s[i] = new_val
                
                # Check if global max dropped
                # Only check if we touched an AP with max value
                if np.any(np.abs(sums[affected]) == current_max) or np.any(new_abs_affected == current_max):
                    global_max = int(np.max(np.abs(sums)))
                    if global_max < current_max:
                        current_max = global_max
                        best_score = current_max
                        best_s = s.copy()
                        
        return best_s, best_score

    # 4. Main Loop with Restarts
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N # Max possible
    
    time_left = time_limit - (time.time() - start_time)
    
    # Run 1: Thue-Morse
    tm = thue_morse(N)
    s1, score1 = optimize_smooth(tm, time_left * 0.5, rng)
    if score1 < best_score_overall:
        best_score_overall = score1
        best_s_overall = s1
        
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 2: Random
        rand1 = random_init(N, rng)
        s2, score2 = optimize_smooth(rand1, time_left * 0.5, rng)
        if score2 < best_score_overall:
            best_score_overall = score2
            best_s_overall = s2
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 3: Random
        rand2 = random_init(N, rng)
        s3, score3 = optimize_smooth(rand2, time_left * 0.8, rng)
        if score3 < best_score_overall:
            best_score_overall = score3
            best_s_overall = s3

    return [int(x) for x in best_s_overall]

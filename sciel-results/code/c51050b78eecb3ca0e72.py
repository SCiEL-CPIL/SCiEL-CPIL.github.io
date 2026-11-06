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
            
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    ap_indices_np = [np.array(lst, dtype=np.int32) for lst in ap_indices]
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    ap_lengths = np.array([len(lst) for lst in ap_indices], dtype=np.int32)
    
    # Precompute for O(1) energy delta computation
    # For each position i, precompute:
    # - aps_containing_np[i]: list of AP indices containing i
    # - For fast energy delta: we need sum of (2*sums[ap] + 2*old_val) for affected APs
    # Actually, delta_E = sum_{ap in affected} [(sums[ap] + delta)^2 - sums[ap]^2]
    #                    = sum_{ap in affected} [2*delta*sums[ap] + delta^2]
    #                    = delta * (2 * sum_{ap in affected} sums[ap]) + delta^2 * len(affected)
    # So we need for each position i: sum of sums[ap] for ap in aps_containing[i]
    # This is a vector that changes as we update sums, so we can't precompute it statically.
    # But we can maintain it incrementally.
    
    # Alternative: just use the affected list and compute delta_E directly
    # delta_E = sum_{ap in affected} [(sums[ap]+delta)^2 - sums[ap]^2]
    #         = delta * 2 * sum_{ap in affected} sums[ap] + delta^2 * len(affected)
    
    # 2. Initialization Functions
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    def perturbed_init(base_s, rng, num_flips):
        s = base_s.copy()
        for _ in range(num_flips):
            i = rng.randint(0, N - 1)
            s[i] = -s[i]
        return s

    # 3. Spin System Monte Carlo Optimization
    # Energy = sum of squared AP sums
    # We minimize energy using heat-bath / Metropolis updates
    def spin_mc_optimize(s_init, time_budget, rng):
        s = s_init.copy()
        
        # Compute initial AP sums
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
        
        # Compute initial energy
        energy = int(np.sum(sums.astype(np.int64) ** 2))
        
        # Track max absolute sum for the objective
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        best_energy = energy
        
        # Temperature schedule
        T = 50.0
        T_min = 0.1
        cooling = 0.999
        
        # Precompute affected AP lengths for each position
        affected_lengths = np.array([len(lst) for lst in aps_containing], dtype=np.int32)
        
        sweep_count = 0
        max_sweeps_no_improve = 100
        sweeps_no_improve = 0
        
        while time.time() - start_time < time_budget:
            # One full sweep: try to update each position once
            for i in range(N):
                if time.time() - start_time > time_budget:
                    break
                
                old_val = int(s[i])
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                if len(affected) == 0:
                    continue
                
                # Compute delta_E = sum_{ap in affected} [(sums[ap]+delta)^2 - sums[ap]^2]
                # = delta * 2 * sum(sums[affected]) + delta^2 * len(affected)
                sum_affected = int(np.sum(sums[affected]))
                delta_E = delta * 2 * sum_affected + delta * delta * len(affected)
                
                if delta_E <= 0:
                    # Always accept if energy decreases or stays same
                    sums[affected] += delta
                    s[i] = new_val
                    energy += delta_E
                else:
                    # Metropolis acceptance
                    if T > T_min:
                        # P(accept) = exp(-delta_E / T)
                        # Use log for numerical stability
                        log_prob = -delta_E / T
                        if log_prob > 0:
                            accept = True
                        else:
                            accept = (rng.random() < np.exp(log_prob))
                        if accept:
                            sums[affected] += delta
                            s[i] = new_val
                            energy += delta_E
                    # If T <= T_min, only accept if delta_E <= 0 (already handled)
            
            sweep_count += 1
            
            # Check for improvement in max absolute sum
            new_max = int(np.max(np.abs(sums)))
            if new_max < current_max:
                current_max = new_max
                if current_max < best_score:
                    best_score = current_max
                    best_s = s.copy()
                    best_energy = energy
                    sweeps_no_improve = 0
                else:
                    sweeps_no_improve += 1
            else:
                sweeps_no_improve += 1
            
            # Cool down temperature
            if sweeps_no_improve % 10 == 0:
                T *= cooling
                if T < T_min:
                    T = T_min
            
            # Occasionally reheat if stuck
            if sweeps_no_improve > max_sweeps_no_improve:
                T = min(50.0, T * 2.0)
                sweeps_no_improve = 0
        
        return best_s, best_score, best_energy

    # 4. Hybrid: Spin MC + Local Search Polish
    def hybrid_optimize(s_init, time_budget, rng):
        # Phase 1: Spin MC to find good basin (60% of time)
        s_mc, score_mc, energy_mc = spin_mc_optimize(s_init, time_budget * 0.6, rng)
        
        # Phase 2: Local search polish on the result (40% of time)
        s_final, score_final = local_search_polish(s_mc, time_budget * 0.4, rng)
        
        return s_final, score_final

    def local_search_polish(s_init, time_budget, rng):
        s = s_init.copy()
        
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
            
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        max_possible_sum = N
        hist = np.zeros(max_possible_sum + 2, dtype=np.int32)
        for val in np.abs(sums):
            hist[val] += 1
        
        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
        
        indices_list = list(range(N))
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = int(s[i])
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                new_sums_affected = sums[affected] + delta
                new_abs_affected = np.abs(new_sums_affected)
                
                if np.any(new_abs_affected > current_max):
                    continue
                
                old_abs_affected = np.abs(sums[affected])
                
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
        
        return best_s, best_score

    # 5. Main Loop with Restarts
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N
    
    time_left = time_limit - (time.time() - start_time)
    
    # Run 1: Thue-Morse with Spin MC (30%)
    tm = thue_morse(N)
    s1, score1 = hybrid_optimize(tm, time_left * 0.30, rng)
    if score1 < best_score_overall:
        best_score_overall = score1
        best_s_overall = s1
        
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 2: Random init with Spin MC (25%)
        rand1 = random_init(N, rng)
        s2, score2 = hybrid_optimize(rand1, time_left * 0.25, rng)
        if score2 < best_score_overall:
            best_score_overall = score2
            best_s_overall = s2
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 3: Perturbed Thue-Morse with Spin MC (20%)
        pert_tm = perturbed_init(tm, rng, 5)
        s3, score3 = hybrid_optimize(pert_tm, time_left * 0.20, rng)
        if score3 < best_score_overall:
            best_score_overall = score3
            best_s_overall = s3
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 4: Another random init (15%)
        rand2 = random_init(N, rng)
        s4, score4 = hybrid_optimize(rand2, time_left * 0.15, rng)
        if score4 < best_score_overall:
            best_score_overall = score4
            best_s_overall = s4
            
    # Final Polish
    time_left = time_limit - (time.time() - start_time)
    if time_left > 0.5 and best_s_overall is not None:
        s_final, score_final = local_search_polish(best_s_overall, time_left * 0.9, rng)
        if score_final < best_score_overall:
            best_score_overall = score_final
            best_s_overall = s_final

    return [int(x) for x in best_s_overall]

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
    
    # Convert to numpy arrays
    ap_indices_np = [np.array(lst, dtype=np.int32) for lst in ap_indices]
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    ap_lengths = np.array([len(lst) for lst in ap_indices], dtype=np.int32)

    # Precompute a flat affected-AP structure for faster per-step updates
    # For each position i, store (start_offset, end_offset) into a flat array of all affected AP indices
    all_affected_flat = []
    affected_offsets = [0] * N
    for i in range(N):
        affected_offsets[i] = len(all_affected_flat)
        all_affected_flat.extend(aps_containing[i])
    affected_offsets.append(len(all_affected_flat))
    all_affected_flat = np.array(all_affected_flat, dtype=np.int32)
    affected_lengths = np.array([affected_offsets[i+1] - affected_offsets[i] for i in range(N)], dtype=np.int32)

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

    # Vectorized sum computation
    def compute_sums(s, ap_indices_np, num_aps):
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
        return sums

    # 3. Core Optimization
    def optimize(s_init, time_budget, rng):
        s = s_init.copy()
        sums = compute_sums(s, ap_indices_np, num_aps)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        # Histogram for O(1) max queries
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
        target_D = max(current_max - 1, 0)
        T = 1.0
        T_min = 0.01
        cooling = 0.95
        stuck_counter = 0
        max_stuck = 300
        perturbation_cooldown = 0

        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                if perturbation_cooldown > 0:
                    perturbation_cooldown -= 1
                    continue
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                # Get affected APs via flat array
                a_start = affected_offsets[i]
                a_end = affected_offsets[i+1]
                affected = all_affected_flat[a_start:a_end]
                
                old_sums_affected = sums[affected]
                new_sums_affected = old_sums_affected + delta
                new_abs_affected = np.abs(new_sums_affected)
                old_abs_affected = np.abs(old_sums_affected)
                
                violates_current_max = np.any(new_abs_affected > current_max)
                
                if violates_current_max:
                    continue
                
                violates_target = np.any(new_abs_affected > target_D)
                
                if not violates_target:
                    # Apply flip
                    changed = old_abs_affected != new_abs_affected
                    if np.any(changed):
                        for j in range(len(changed)):
                            if changed[j]:
                                hist[old_abs_affected[j]] -= 1
                                hist[new_abs_affected[j]] += 1
                    sums[affected] = new_sums_affected
                    s[i] = new_val
                    stuck_counter = 0
                    
                    new_global_max = get_max_from_hist()
                    if new_global_max < current_max:
                        current_max = new_global_max
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                        target_D = max(current_max - 1, 0)
                        T = 1.0
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    # Relaxed move
                    max_violation = int(np.max(new_abs_affected - target_D))
                    p_accept = 0.03 + 0.12 * (stuck_counter / max_stuck) - 0.008 * max_violation
                    p_accept = max(0.0, min(0.4, p_accept))
                    if rng.random() < p_accept:
                        changed = old_abs_affected != new_abs_affected
                        if np.any(changed):
                            for j in range(len(changed)):
                                if changed[j]:
                                    hist[old_abs_affected[j]] -= 1
                                    hist[new_abs_affected[j]] += 1
                        sums[affected] = new_sums_affected
                        s[i] = new_val
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                
                if stuck_counter >= max_stuck:
                    num_perturb = rng.randint(2, 4)
                    perturbation_indices = rng.sample(range(N), num_perturb)
                    
                    s_backup = s.copy()
                    sums_backup = sums.copy()
                    hist_backup = hist.copy()
                    
                    accepted_any = False
                    for pi in perturbation_indices:
                        old_val = s[pi]
                        delta = -2 * old_val
                        new_val = -old_val
                        a_start = affected_offsets[pi]
                        a_end = affected_offsets[pi+1]
                        affected = all_affected_flat[a_start:a_end]
                        new_sums_affected = sums[affected] + delta
                        new_abs_affected = np.abs(new_sums_affected)
                        
                        if not np.any(new_abs_affected > current_max):
                            old_abs_affected = np.abs(sums[affected])
                            changed = old_abs_affected != new_abs_affected
                            if np.any(changed):
                                for j in range(len(changed)):
                                    if changed[j]:
                                        hist[old_abs_affected[j]] -= 1
                                        hist[new_abs_affected[j]] += 1
                            sums[affected] = new_sums_affected
                            s[pi] = new_val
                            accepted_any = True
                    
                    if accepted_any:
                        new_global_max = get_max_from_hist()
                        if new_global_max < current_max:
                            current_max = new_global_max
                            if current_max < best_score:
                                best_score = current_max
                                best_s = s.copy()
                            target_D = max(current_max - 1, 0)
                            stuck_counter = 0
                            T = 1.0
                        else:
                            s = s_backup
                            sums = sums_backup
                            hist = hist_backup
                            stuck_counter = 0
                            T *= cooling
                            if T < T_min:
                                T = T_min
                    else:
                        stuck_counter = 0
                        T *= cooling
                        if T < T_min:
                            T = T_min
                    perturbation_cooldown = 8

        return best_s, best_score

    # 4. Main Loop
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N
    
    time_left = time_limit - (time.time() - start_time)
    
    # Run 1: Thue-Morse
    tm = thue_morse(N)
    s1, score1 = optimize(tm, time_left * 0.25, rng)
    if score1 < best_score_overall:
        best_score_overall = score1
        best_s_overall = s1
        
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        rand1 = random_init(N, rng)
        s2, score2 = optimize(rand1, time_left * 0.25, rng)
        if score2 < best_score_overall:
            best_score_overall = score2
            best_s_overall = s2
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        pert_tm = perturbed_init(tm, rng, 3)
        s3, score3 = optimize(pert_tm, time_left * 0.25, rng)
        if score3 < best_score_overall:
            best_score_overall = score3
            best_s_overall = s3
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        rand2 = random_init(N, rng)
        s4, score4 = optimize(rand2, time_left * 0.25, rng)
        if score4 < best_score_overall:
            best_score_overall = score4
            best_s_overall = s4

    # Final Polish
    time_left = time_limit - (time.time() - start_time)
    if time_left > 0.5 and best_s_overall is not None:
        s_final, score_final = optimize(best_s_overall, time_left * 0.95, rng)
        if score_final < best_score_overall:
            best_score_overall = score_final
            best_s_overall = s_final

    return [int(x) for x in best_s_overall]

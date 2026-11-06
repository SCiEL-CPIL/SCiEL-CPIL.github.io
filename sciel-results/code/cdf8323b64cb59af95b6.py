import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin

    N = 256

    # 1. Precompute ALL APs
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
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]

    # 2. Initialization
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    # 3. Optimization with plateau-escape via tie-breaking
    def optimize(s_init, time_budget, rng, allow_tie_break=True):
        s = s_init.copy()
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(s[idx_list])
            
        abs_sums = np.abs(sums)
        max_val = int(np.max(abs_sums))
        counts = np.bincount(abs_sums, minlength=max_val + 2)
        current_max = max_val
        current_max_count = int(counts[current_max])
        
        best_s = s.copy()
        best_score = current_max
        best_max_count = current_max_count
        
        indices_list = list(range(N))
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                old_sums = sums[affected].copy()
                sums[affected] += delta
                
                max_new_affected = int(np.max(np.abs(sums[affected])))
                if max_new_affected > current_max:
                    sums[affected] = old_sums
                    continue
                
                old_abs = np.abs(old_sums)
                new_abs = np.abs(sums[affected])
                
                changed_mask = old_abs != new_abs
                if np.any(changed_mask):
                    changed_old = old_abs[changed_mask]
                    changed_new = new_abs[changed_mask]
                    
                    np.add.at(counts, changed_old, -1)
                    np.add.at(counts, changed_new, 1)
                    
                    if counts[current_max] == 0:
                        new_max = current_max
                        while new_max > 0 and counts[new_max] == 0:
                            new_max -= 1
                        current_max = new_max
                        current_max_count = int(counts[current_max])
                        
                        if current_max < best_score or (current_max == best_score and current_max_count < best_max_count):
                            best_score = current_max
                            best_max_count = current_max_count
                            best_s = s.copy()
                    else:
                        # Max unchanged; accept if tie-break allowed and count of max decreased
                        new_max_count = int(counts[current_max])
                        if allow_tie_break and new_max_count < current_max_count:
                            current_max_count = new_max_count
                            if current_max < best_score or (current_max == best_score and current_max_count < best_max_count):
                                best_score = current_max
                                best_max_count = current_max_count
                                best_s = s.copy()
                        else:
                            # Revert: max unchanged and count not improved
                            sums[affected] = old_sums
                            np.add.at(counts, changed_old, 1)
                            np.add.at(counts, changed_new, -1)
                            continue
                else:
                    # No abs sum changed; accept unconditionally (neutral move)
                    pass
                
                s[i] = new_val

        return best_s, best_score

    rng = random.Random(42)
    
    # Run 1: Thue-Morse with tie-break
    tm = thue_morse(N)
    s1, score1 = optimize(tm, time_limit * 0.35, rng, allow_tie_break=True)
    best_s = s1
    best_score = score1

    # Run 2: Random init with tie-break
    if time.time() - start_time < time_limit * 0.7:
        rand_init = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = optimize(rand_init, time_limit * 0.25, rng, allow_tie_break=True)
        if score2 < best_score:
            best_score = score2
            best_s = s2

    # Run 3: Polish pass — strict max-reduction only for final confidence
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        s3, score3 = optimize(best_s, remaining - 0.5, rng, allow_tie_break=False)
        if score3 < best_score:
            best_score = score3
            best_s = s3

    return [int(x) for x in best_s]

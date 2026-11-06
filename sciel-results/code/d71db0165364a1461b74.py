import numpy as np
import random
import time

def construct():
    n = 128
    target_lo = 128
    target_hi = 256
    start_time = time.time()
    
    def check(vec):
        conv = np.convolve(vec, vec)
        target_counts = conv[target_lo:target_hi+1]
        s_vals = np.arange(target_lo, target_hi+1)
        D_s = np.zeros(len(s_vals), dtype=int)
        even_indices = np.where(s_vals % 2 == 0)[0]
        if len(even_indices) > 0:
            s_even = s_vals[even_indices]
            mid_indices = s_even // 2
            valid_mids = (mid_indices >= 0) & (mid_indices <= 128)
            D_s[even_indices[valid_mids]] = vec[mid_indices[valid_mids]]
        unordered_counts = (target_counts + D_s) // 2
        valid = np.all(unordered_counts > 0)
        peak = np.max(unordered_counts) if valid else 0
        return valid, peak, unordered_counts

    def repair(vec):
        # Greedy repair
        valid, peak, counts = check(vec)
        while not valid and time.time() - start_time < 50:
            uncovered = np.where(counts == 0)[0]
            best_elem = -1
            best_gain = 0
            # Sample candidates to speed up
            candidates = list(range(129))
            random.shuffle(candidates)
            for e in candidates:
                if vec[e] == 1:
                    continue
                vec[e] = 1
                v, p, c = check(vec)
                gain = np.sum((c > 0) & (counts == 0))
                vec[e] = 0
                if gain > best_gain:
                    best_gain = gain
                    best_elem = e
            if best_elem == -1:
                break
            vec[best_elem] = 1
            valid, peak, counts = check(vec)
        return vec

    best_peak = float('inf')
    best_vec = None
    
    # Try different initial sizes
    for size in [45, 55, 65, 75]:
        for restart in range(3):
            if time.time() - start_time > 45:
                break
            
            vec = np.zeros(129, dtype=int)
            indices = random.sample(range(129), size)
            vec[indices] = 1
            
            vec = repair(vec)
            
            valid, peak, _ = check(vec)
            if not valid:
                continue
            
            current_peak = peak
            
            # Local Search
            improved = True
            passes = 0
            while improved and time.time() - start_time < 55:
                improved = False
                passes += 1
                
                # Try a random subset of swaps
                # Instead of all N^2, try N * 10 random swaps
                candidates_out = [i for i in range(129) if vec[i] == 1]
                candidates_in = [i for i in range(129) if vec[i] == 0]
                
                if not candidates_out or not candidates_in:
                    break
                
                num_swaps = min(500, len(candidates_out) * len(candidates_in))
                
                for _ in range(num_swaps):
                    e_out = random.choice(candidates_out)
                    e_in = random.choice(candidates_in)
                    
                    vec[e_out] = 0
                    vec[e_in] = 1
                    
                    v, p, _ = check(vec)
                    
                    if v and p < current_peak:
                        current_peak = p
                        improved = True
                        # Update candidate lists? 
                        # It's easier to just rebuild or check membership
                        candidates_out.remove(e_out)
                        candidates_out.append(e_in)
                        candidates_in.remove(e_in)
                        candidates_in.append(e_out)
                        break
                    else:
                        vec[e_out] = 1
                        vec[e_in] = 0

                # Try removals
                if not improved and size > 35:
                    candidates_out = [i for i in range(129) if vec[i] == 1]
                    random.shuffle(candidates_out)
                    for e_out in candidates_out[:20]: # Try 20 removals
                        vec[e_out] = 0
                        v, p, _ = check(vec)
                        if v and p < current_peak:
                            current_peak = p
                            improved = True
                            size -= 1
                            break
                        else:
                            vec[e_out] = 1

            if peak < best_peak:
                best_peak = peak
                best_vec = vec.copy()

    if best_vec is None:
        vec = np.ones(129, dtype=int)
        best_vec = vec
        
    return list(np.where(best_vec == 1)[0])

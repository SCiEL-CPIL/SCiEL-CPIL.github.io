import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = 129
    
    # Precompute pair_target[a][b] for a<=b
    # target index = a + b - 128
    pair_target = np.full((n+1, n+1), -1, dtype=np.int32)
    for a in range(n+1):
        for b in range(a, n+1):
            s = a + b
            if target_lo <= s <= target_hi:
                pair_target[a][b] = s - target_lo

    def get_counts(A_set):
        counts = np.zeros(num_targets, dtype=np.int32)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                t = pair_target[a][b]
                if t >= 0:
                    counts[t] += 1
        return counts

    def update_counts_remove(counts, x, A_set):
        # x pairs with a in A_set where a <= x
        for a in A_set:
            if a <= x:
                t = pair_target[a][x]
                if t >= 0:
                    counts[t] -= 1
        # Note: if x in A_set, the pair (x,x) is handled when a=x
        # But we must ensure we don't double count or miss.
        # The loop `for a in A_set: if a <= x` covers all a.
        # If a=x, it covers (x,x).
        # If a<x, it covers (a,x).
        # This is correct for unordered pairs.
        
    def update_counts_add(counts, x, A_set):
        for a in A_set:
            if a <= x:
                t = pair_target[a][x]
                if t >= 0:
                    counts[t] += 1
        # Add self pair
        t = pair_target[x][x]
        if t >= 0:
            counts[t] += 1

    def is_valid(counts):
        return np.min(counts) > 0

    def peak(counts):
        return int(np.max(counts))

    def score_state(counts):
        if not is_valid(counts):
            return (1, peak(counts)) # Invalid, prioritize validity
        return (0, peak(counts))

    best_A = None
    best_peak = float('inf')
    
    random.seed(42)
    
    # Try different sizes
    for k in range(16, 25):
        for restart in range(20):
            if time.time() > deadline:
                return best_A if best_A else list(range(64, 129))
            
            # Random Init
            A_set = set(random.sample(range(n+1), k))
            counts = get_counts(A_set)
            
            # Local Search
            cur_peak = peak(counts)
            cur_sum_sq = int(np.sum(counts**2))
            
            # If already valid and peak is low, keep it
            if is_valid(counts) and cur_peak < best_peak:
                best_peak = cur_peak
                best_A = list(A_set)
                
            # Greedy/Local Search
            for it in range(5000):
                if time.time() > deadline:
                    break
                    
                improved = False
                
                # Try Swaps
                # Sample a few candidates
                candidates = list(A_set)
                random.shuffle(candidates)
                
                for x_out in candidates[:5]: # Limit to speed up
                    # Try to swap with a random x_in
                    for _ in range(10):
                        x_in = random.randint(0, n)
                        if x_in in A_set or x_in == x_out:
                            continue
                        
                        # Simulate Swap
                        old_counts = counts.copy()
                        
                        # Remove x_out
                        update_counts_remove(counts, x_out, A_set)
                        # Add x_in
                        update_counts_add(counts, x_in, A_set)
                        
                        if is_valid(counts):
                            new_peak = peak(counts)
                            new_sum_sq = int(np.sum(counts**2))
                            
                            # Accept if peak is lower, or peak same and sum_sq lower
                            if new_peak < cur_peak or (new_peak == cur_peak and new_sum_sq < cur_sum_sq):
                                # Commit
                                A_set.remove(x_out)
                                A_set.add(x_in)
                                cur_peak = new_peak
                                cur_sum_sq = new_sum_sq
                                improved = True
                                break
                        else:
                            # Revert
                            counts[:] = old_counts
                            
                        # Revert if not improved
                        if not improved:
                            counts[:] = old_counts
                            
                    if improved:
                        break
                        
                if not improved:
                    # Try Additions/Removals if stuck?
                    # Or just break
                    break
                    
            if is_valid(counts) and cur_peak < best_peak:
                best_peak = cur_peak
                best_A = list(A_set)
                
    if best_A is None:
        return list(range(64, 129))
        
    return best_A

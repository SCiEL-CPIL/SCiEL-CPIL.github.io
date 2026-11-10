import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    def compute_counts(v):
        """v: indicator vector of A (length 129). Returns counts for targets [128..256]."""
        # Convolution of v with v gives ordered pair counts
        # conv[k] = sum_{i+j=k} v[i]*v[j]
        conv = np.convolve(v, v, mode='full')
        # conv has length 257 (indices 0..256)
        # Target s corresponds to conv[s]
        diag = np.zeros(num_targets, dtype=np.int32)
        for idx, s in enumerate(range(target_lo, target_hi + 1)):
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= n:
                    diag[idx] = v[mid]
        ordered = conv[target_lo:target_hi + 1].astype(np.int32)
        unordered = (ordered + diag) // 2
        return unordered
    
    def is_complete(counts):
        return bool(np.all(counts > 0))
    
    def peak_of(counts):
        return int(np.max(counts))
    
    def evaluate(A_set):
        v = np.zeros(n + 1, dtype=np.int32)
        for x in A_set:
            v[x] = 1
        counts = compute_counts(v)
        if not is_complete(counts):
            return None, None
        return A_set, peak_of(counts)
    
    # Greedy construction: start empty, add elements to cover targets
    # Priority: cover targets with fewest current representations first
    
    def greedy_construct():
        A = set()
        v = np.zeros(n + 1, dtype=np.int32)
        counts = np.zeros(num_targets, dtype=np.int32)
        
        # Precompute: for each element x, which targets does it help with?
        # When we add x, it forms pairs (a,x) for all a in A with a<=x, plus (x,x)
        
        for step in range(200):
            if time.time() > deadline - 30:
                break
            # Find the target with minimum count (most urgent)
            if not is_complete(counts):
                # Find targets with 0 count
                zero_targets = np.where(counts == 0)[0]
                if len(zero_targets) > 0:
                    # Pick the first zero target
                    target_idx = zero_targets[0]
                    target_s = target_lo + target_idx
                else:
                    break
            else:
                break
            
            # Find the element x that covers the most uncovered targets
            best_x = None
            best_coverage = -1
            for x in range(n + 1):
                if x in A:
                    continue
                # If we add x, which new targets get covered?
                # x pairs with all a in A where a <= x, plus x itself
                new_targets = set()
                for a in A:
                    if a <= x:
                        s = a + x
                        if target_lo <= s <= target_hi:
                            new_targets.add(s - target_lo)
                s = x + x
                if target_lo <= s <= target_hi:
                    new_targets.add(s - target_lo)
                
                # Count how many of these are currently 0
                coverage = sum(1 for t in new_targets if counts[t] == 0)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_x = x
            
            if best_x is None or best_coverage <= 0:
                break
            
            A.add(best_x)
            v[best_x] = 1
            # Update counts
            for a in A:
                if a <= best_x:
                    s = a + best_x
                    if target_lo <= s <= target_hi:
                        counts[s - target_lo] += 1
            s = best_x + best_x
            if target_lo <= s <= target_hi:
                counts[s - target_lo] += 1
        
        return A, counts
    
    # Local search using vectorized evaluation
    def local_search(A_init, max_iter=50000):
        A = set(A_init)
        v = np.zeros(n + 1, dtype=np.int32)
        for x in A:
            v[x] = 1
        counts = compute_counts(v)
        
        if not is_complete(counts):
            return A, peak_of(counts) if counts.size > 0 else 999
        
        cur_peak = peak_of(counts)
        A_list = sorted(A)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try removals
            for x in A_list[:]:
                v[x] = 0
                new_counts = compute_counts(v)
                v[x] = 1
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A.discard(x)
                        A_list = [a for a in A_list if a != x]
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try additions
            for x in range(n + 1):
                if x in A:
                    continue
                v[x] = 1
                new_counts = compute_counts(v)
                v[x] = 0
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A.add(x)
                        A_list.append(x)
                        A_list.sort()
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swaps (sampled)
            found_swap = False
            num_swaps = min(15, len(A_list))
            for _ in range(num_swaps):
                if time.time() > deadline - 10:
                    break
                idx = random.randint(0, len(A_list) - 1)
                x_out = A_list[idx]
                x_in = random.randint(0, n)
                if x_in in A or x_in == x_out:
                    continue
                
                v[x_out] = 0
                v[x_in] = 1
                new_counts = compute_counts(v)
                v[x_in] = 0
                v[x_out] = 1
                
                if is_complete(new_counts):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A.discard(x_out)
                        A.add(x_in)
                        A_list[idx] = x_in
                        A_list.sort()
                        counts = new_counts
                        cur_peak = new_peak
                        found_swap = True
                        break
            
            if not found_swap:
                break
        
        return A, cur_peak
    
    # Run greedy construction first
    random.seed(42)
    
    best_A = None
    best_peak = float('inf')
    
    # Greedy
    A_greedy, counts_greedy = greedy_construct()
    if is_complete(counts_greedy):
        peak_greedy = peak_of(counts_greedy)
        A_ls, peak_ls = local_search(A_greedy)
        if peak_ls < best_peak:
            best_A = A_ls
            best_peak = peak_ls
    
    # Structured initializations
    initializations = [
        list(range(64, 129)),
        list(range(0, 129, 2)),
        list(range(32, 129, 2)),
        list(range(0, 65)) + list(range(96, 129)),
    ]
    
    for init_A in initializations:
        if time.time() > deadline - 20:
            break
        A_ls, peak_ls = local_search(init_A, max_iter=30000)
        if peak_ls < best_peak:
            best_A = A_ls
            best_peak = peak_ls
    
    # Random restarts
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        k = random.randint(50, 80)
        A_rand = set(random.sample(range(n + 1), k))
        A_ls, peak_ls = local_search(A_rand, max_iter=20000)
        if peak_ls < best_peak:
            best_A = A_ls
            best_peak = peak_ls
    
    # Final validation
    if best_A is None:
        return list(range(64, 129))
    
    # Verify
    v = np.zeros(n + 1, dtype=np.int32)
    for x in best_A:
        v[x] = 1
    final_counts = compute_counts(v)
    if not is_complete(final_counts):
        return list(range(64, 129))
    
    return list(best_A)

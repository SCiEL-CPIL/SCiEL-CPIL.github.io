import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    # Precompute pairs for each target for quick lookup in local search
    target_pairs = []
    for i in range(num_targets):
        t = target_lo + i
        pairs = []
        for a in range(t // 2 + 1):
            b = t - a
            if 0 <= a <= n and 0 <= b <= n:
                pairs.append((a, b))
        target_pairs.append(pairs)
        
    def compute_counts(A_set):
        f = np.zeros(257, dtype=np.float64)
        for a in A_set:
            f[a] = 1.0
        c = np.convolve(f, f)
        counts = np.zeros(num_targets, dtype=np.int64)
        for i in range(num_targets):
            t = target_lo + i
            ordered = int(c[t])
            if t % 2 == 0:
                mid = t // 2
                if mid in A_set:
                    counts[i] = (ordered + 1) // 2
                else:
                    counts[i] = ordered // 2
            else:
                counts[i] = ordered // 2
        return counts

    def is_valid(A_set):
        counts = compute_counts(A_set)
        return int(np.min(counts)) >= 1

    def get_peak(A_set):
        counts = compute_counts(A_set)
        return int(np.max(counts))

    # Strategy: Start with a valid set, then minimize peak.
    # A simple valid set: All numbers from 64 to 128.
    A_current = set(range(64, 129))
    if not is_valid(A_current):
        A_current = set(range(0, 129))
        
    best_A = set(A_current)
    best_peak = get_peak(best_A)
    counts = compute_counts(A_current)
    
    max_iters = 5000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        max_c = int(np.max(counts))
        if max_c == 1:
            break
        
        max_idxs = np.where(counts == max_c)[0]
        t_idx = max_idxs[random.randint(0, len(max_idxs)-1)]
        t_val = target_lo + t_idx
        
        pairs_in_A = []
        for a, b in target_pairs[t_idx]:
            if a in A_current and b in A_current:
                pairs_in_A.append((a, b))
                
        if not pairs_in_A:
            continue
            
        a, b = pairs_in_A[random.randint(0, len(pairs_in_A)-1)]
        
        if a != b:
            A_test = set(A_current)
            A_test.discard(a)
            if is_valid(A_test):
                new_counts = compute_counts(A_test)
                new_peak = int(np.max(new_counts))
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = A_test
                    A_current = A_test
                    counts = new_counts
                    continue
                elif new_peak == best_peak:
                    A_current = A_test
                    counts = new_counts
                    continue

        if a != b:
            A_test = set(A_current)
            A_test.discard(b)
            if is_valid(A_test):
                new_counts = compute_counts(A_test)
                new_peak = int(np.max(new_counts))
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = A_test
                    A_current = A_test
                    counts = new_counts
                    continue
                elif new_peak == best_peak:
                    A_current = A_test
                    counts = new_counts
                    continue

        x = random.randint(0, n)
        if x not in A_current and x != a:
            A_test = set(A_current)
            A_test.discard(a)
            A_test.add(x)
            if is_valid(A_test):
                new_counts = compute_counts(A_test)
                new_peak = int(np.max(new_counts))
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = A_test
                    A_current = A_test
                    counts = new_counts
                    continue
                elif new_peak == best_peak:
                    if random.random() < 0.5:
                        A_current = A_test
                        counts = new_counts
                        continue

    if not is_valid(best_A):
        return list(range(64, 129))
        
    return list(best_A)

import time
import random
import math
import numpy as np

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def calc_score_np(A):
    """
    Fast score computation using NumPy.
    """
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    # Build a rich pool of candidate numbers
    pool_set = set()
    for i in range(1, 301):
        pool_set.add(i)
    # Smooth numbers: products of small primes
    small_primes = [2, 3, 5, 7, 11, 13]
    def gen_smooth(max_val, primes, depth=0, current=1):
        if current > max_val:
            return
        pool_set.add(current)
        for i in range(depth, len(primes)):
            if current * primes[i] <= max_val:
                gen_smooth(max_val, primes, i + 1, current * primes[i])
    gen_smooth(8000, small_primes)
    # Add structured numbers: multiples of small bases
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 30, 36, 40, 48, 50, 60, 72, 80, 90, 100]:
        for k in range(1, 300):
            v = base * k
            if v <= 8000:
                pool_set.add(v)
    # Add some geometric progressions
    for base in [2, 3, 4]:
        v = base
        while v <= 8000:
            pool_set.add(v)
            v *= base
    
    pool = sorted(pool_set)
    pool_np = np.array(pool, dtype=np.int64)
    
    best_set = None
    best_score = float('inf')
    
    def greedy_construct(seed):
        """Greedily build a set of 20 elements starting from seed."""
        A = sorted(list(set(seed)))
        if len(A) < 5:
            return None
        
        while len(A) < 20:
            best_elem = None
            best_sc = float('inf')
            candidates = [c for c in pool if c not in A]
            random.shuffle(candidates)
            # Limit candidates for speed
            sample_size = min(400, len(candidates))
            for c in candidates[:sample_size]:
                new_A = A + [c]
                sc = calc_score_np(new_A)
                if sc < best_sc:
                    best_sc = sc
                    best_elem = c
            if best_elem is not None:
                A.append(best_elem)
                A.sort()
            else:
                for c in pool:
                    if c not in A:
                        A.append(c)
                        A.sort()
                        break
        return A
    
    def local_search_best_improve(A, time_limit):
        """Best-improvement 1-swap local search with NumPy acceleration."""
        A = sorted(A[:])
        current_score = calc_score_np(A)
        best_local = A[:]
        best_local_score = current_score
        
        # Precompute: for each position, which pool values to try
        # Try values close to the current value first (locality)
        while time.time() < time_limit:
            improved = False
            for i in range(len(A)):
                old_val = A[i]
                # Generate candidate values: nearby values in pool
                idx_in_pool = pool.index(old_val) if old_val in pool else 0
                # Try a window of candidates around the current value
                start_idx = max(0, idx_in_pool - 50)
                end_idx = min(len(pool), idx_in_pool + 50)
                candidates_nearby = pool[start_idx:end_idx]
                # Also add some random candidates
                random_candidates = random.sample(pool, min(50, len(pool)))
                all_candidates = list(set(candidates_nearby + random_candidates))
                
                for new_val in all_candidates:
                    if new_val == old_val or new_val in A:
                        continue
                    A[i] = new_val
                    A.sort()
                    new_score = calc_score_np(A)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_local_score:
                            best_local_score = current_score
                            best_local = A[:]
                    else:
                        A[i] = old_val
                        A.sort()
                if improved:
                    break
            if not improved:
                break
        return best_local, best_local_score
    
    # Multiple greedy starts with different seeds
    seeds = [
        [1, 2, 3, 4, 5],
        [1, 2, 4, 8, 16],
        [1, 2, 3, 6, 12],
        [1, 2, 4, 6, 8],
        [2, 3, 4, 6, 9],
        [1, 3, 4, 6, 8],
        [1, 2, 3, 4, 6],
        [1, 2, 4, 6, 12],
        [1, 2, 3, 5, 10],
        [2, 4, 6, 8, 10],
        [1, 2, 4, 8, 12],
        [1, 3, 6, 9, 12],
    ]
    
    # Run greedy construction for each seed
    candidates = []
    for seed in seeds:
        if time.time() > deadline - 12:
            break
        g_set = greedy_construct(seed)
        if g_set:
            g_score = calc_score_np(g_set)
            candidates.append((g_set, g_score))
    
    # Sort by score, take top few for local search
    candidates.sort(key=lambda x: x[1])
    
    # Local search on the best candidates
    for g_set, g_score in candidates[:4]:
        if time.time() > deadline - 5:
            break
        ls_set, ls_score = local_search_best_improve(g_set, deadline - 5)
        if ls_score < best_score:
            best_score = ls_score
            best_set = ls_set[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = calc_score(best_set)
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(res)
    print(calc_score(res))

import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    def sieve(n):
        # Use NumPy for speed
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return [int(p) for p in np.nonzero(is_prime)[0]]

    primes = sieve(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return primes

    # fronts[i] is a list of (gap, length, prev_idx)
    # Sorted by gap ascending. Lengths strictly increasing.
    fronts = [None] * n
    fronts[0] = [(0, 1, -1)]
    
    best_len = 1
    best_end = 0

    # Precompute primes array for fast access
    p_arr = primes

    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break
        
        pi = p_arr[i]
        candidates = [(0, 1, -1)]
        
        for j in range(i):
            front_j = fronts[j]
            if not front_j:
                continue
            
            gap = pi - p_arr[j]
            
            # Binary search for max length with gap_j <= gap
            # front_j is sorted by gap
            lo, hi = 0, len(front_j)
            while lo < hi:
                mid = (lo + hi) // 2
                if front_j[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            
            if lo > 0:
                last_gap, length, prev_idx = front_j[lo - 1]
                candidates.append((gap, length + 1, j))

        # Build Pareto front
        # Sort by gap asc, length desc
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        front_i = []
        max_len_so_far = 0
        for gap, length, prev_idx in candidates:
            if length > max_len_so_far:
                front_i.append((gap, length, prev_idx))
                max_len_so_far = length
        
        fronts[i] = front_i
        
        if front_i:
            # The last element has the max length
            if front_i[-1][1] > best_len:
                best_len = front_i[-1][1]
                best_end = i

    # Reconstruct
    if best_len <= 1:
        return [primes[best_end]]

    seq = []
    idx = best_end
    curr_len = best_len
    
    while curr_len > 0:
        # Find state with length curr_len in fronts[idx]
        # Since lengths are unique and increasing, we can just find it
        # But we need the prev_idx
        target_state = None
        for gap, length, p_idx in fronts[idx]:
            if length == curr_len:
                target_state = (gap, length, p_idx)
                break
        
        if target_state is None:
            # Fallback: use last
            target_state = fronts[idx][-1]
            
        seq.append(primes[idx])
        idx = target_state[2]
        curr_len -= 1
        
        if idx == -1:
            break

    seq.reverse()
    
    # Final check
    if not seq:
        return [primes[0]]
        
    return seq

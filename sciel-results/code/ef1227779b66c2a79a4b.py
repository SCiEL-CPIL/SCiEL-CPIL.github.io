import time
import numpy as np
from math import isqrt
from collections import defaultdict

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2).
    1 counts as powerful."""
    if n == 1:
        return True
    if n < 1:
        return False
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            cnt = 0
            while x % d == 0:
                x //= d
                cnt += 1
            if cnt < 2:
                return False
        d += 1
    if x > 1:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    # Generate powerful numbers up to L.
    # Every powerful number n can be written as a^2 * b^3 for some positive integers a, b.
    
    L = 2 * 10**8  # Upper bound for individual powerful numbers
    
    # Generate candidates as a^2 * b^3 <= L
    powerful_set = set()
    
    max_b = int(L ** (1.0/3.0)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    candidates = sorted(powerful_set)
    
    # Filter to keep only actual powerful numbers
    powerful_list = []
    for n in candidates:
        if _is_powerful(n):
            powerful_list.append(n)
        if time.time() > deadline - 40:
            break
    
    print(f"Generated {len(powerful_list)} powerful numbers up to {L}")
    
    if not powerful_list:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    powerful_set_list = set(powerful_list)
    pw = np.array(powerful_list, dtype=np.int64)
    n_pw = len(pw)
    
    # Find N maximizing pair count using chunked bincount approach.
    # We compute ordered pair counts, then convert to unordered.
    
    # Use a dict for counts since N can be up to 2*L = 4*10^8
    # But we'll use numpy bincount on a reasonable range and process in chunks.
    
    # Actually, let's use a smarter approach: 
    # For each a in powerful_list, the possible N values are a + b for b in powerful_list, b >= a.
    # We want to find the N with the most such b's.
    
    # Use numpy to compute sums in chunks and use a dictionary for accumulation.
    
    # For memory efficiency, process row by row or in small chunks.
    # n_pw could be ~15k for L=2*10^8, so n_pw^2 = 225M. 
    # A float64 array of 225M is 1.8GB - too much.
    # Process in chunks of 1000 rows: 1000 * 15000 = 15M entries = 120MB. OK.
    
    chunk_size = 1000
    sum_counts = defaultdict(int)
    
    for i_start in range(0, n_pw, chunk_size):
        i_end = min(i_start + chunk_size, n_pw)
        chunk = pw[i_start:i_end]
        
        # For each row i in this chunk, we want sums with columns j >= i
        # to avoid double counting. But for simplicity, compute all and divide later.
        # Actually, let's compute all pairs and handle a<=b in the final count.
        
        sums = chunk[:, None] + pw[None, :]
        flat_sums = sums.ravel()
        
        # Use numpy to count
        # Filter to valid range
        mask = (flat_sums >= 2) & (flat_sums <= 2 * L)
        flat_sums = flat_sums[mask]
        
        if len(flat_sums) > 0:
            unique, cnts = np.unique(flat_sums, return_counts=True)
            for idx in range(len(unique)):
                sum_counts[int(unique[idx])] += int(cnts[idx])
        
        if time.time() > deadline - 15:
            break
    
    # Find best N from sum_counts
    # sum_counts[N] = number of ordered pairs (a,b) with a+b=N, both powerful.
    # For unordered pairs with a<=b:
    #   If N is odd: unordered = sum_counts[N] // 2
    #   If N is even: unordered = (sum_counts[N] + (1 if N//2 in powerful_set_list else 0)) // 2
    
    if not sum_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = 2
    best_unordered = 0
    
    for N, ordered in sum_counts.items():
        if N % 2 == 0:
            delta = 1 if (N // 2) in powerful_set_list else 0
            unordered = (ordered + delta) // 2
        else:
            unordered = ordered // 2
        
        if unordered > best_unordered:
            best_unordered = unordered
            best_N = N
    
    print(f"Best N = {best_N}, unordered count = {best_unordered}")
    
    # Generate pairs for best_N
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set_list:
            pairs.append([a, b])
    
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}

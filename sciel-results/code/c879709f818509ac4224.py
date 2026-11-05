"""Stronger algorithm for longest increasing prime subsequence with nondecreasing gaps."""
import time
import numpy as np

def _sieve(n):
    """Return a numpy boolean array where arr[i] is True iff i is prime."""
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
    return is_prime

def _get_primes(is_prime, limit):
    """Extract list of primes up to limit from sieve."""
    return [int(p) for p in range(2, limit + 1) if is_prime[p]]

def _dp_longest(primes, gap_limit, deadline, start_time):
    n = len(primes)
    if n == 0:
        return [], []
    
    # best[i][g] = max length of nondecreasing-gap subsequence ending at primes[i] with last gap = g
    best = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    prefix_best = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    prefix_best_idx = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    parent_i = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    parent_g = [np.zeros(gap_limit + 1, dtype=np.int32) for _ in range(n)]
    
    for i in range(n):
        best[i][0] = 1
        prefix_best[i][0] = 1
        prefix_best_idx[i][0] = 0
        parent_i[i][0] = -1
        parent_g[i][0] = -1
    
    prime_to_idx = {primes[i]: i for i in range(n)}
    
    for i in range(1, n):
        for d in range(1, gap_limit + 1):
            prev_val = primes[i] - d
            if prev_val < 2 or prev_val not in prime_to_idx:
                continue
            j = prime_to_idx[prev_val]
            candidate = prefix_best[j][d] + 1
            if candidate > best[i][d]:
                best[i][d] = candidate
                parent_i[i][d] = j
                parent_g[i][d] = prefix_best_idx[j][d]
        
        for g in range(1, gap_limit + 1):
            if best[i][g] > prefix_best[i][g-1]:
                prefix_best[i][g] = best[i][g]
                prefix_best_idx[i][g] = g
            else:
                prefix_best[i][g] = prefix_best[i][g-1]
                prefix_best_idx[i][g] = prefix_best_idx[i][g-1]
        
        if (i % 100 == 0) and time.time() > deadline:
            break
    
    max_len = 0
    best_i = -1
    best_g = -1
    for i in range(n):
        for g in range(1, gap_limit + 1):
            if best[i][g] > max_len:
                max_len = best[i][g]
                best_i = i
                best_g = g
        if best[i][0] > max_len:
            max_len = best[i][0]
            best_i = i
            best_g = 0
    
    if best_i < 0:
        return [primes[0]], []
    
    result = []
    cur_i = best_i
    cur_g = best_g
    while cur_i >= 0:
        result.append(primes[cur_i])
        if cur_g == 0:
            break
        cur_i = parent_i[cur_i][cur_g]
        cur_g = parent_g[cur_i][cur_g]
    
    result.reverse()
    return result, [result[i+1] - result[i] for i in range(len(result)-1)] if len(result) > 1 else []

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    limit = 10000
    is_prime = _sieve(limit)
    primes = _get_primes(is_prime, limit)
    
    if not primes:
        return []
    
    # Try DP approach
    if time.time() < deadline:
        result, gaps = _dp_longest(primes, 10000, deadline, start_time)
        if result and len(result) > 0:
            # Verify the result
            if all(result[i+1] - result[i] >= result[i] - result[i-1] for i in range(1, len(result)-1)):
                if all(2 <= p <= 10000 and is_prime[p] for p in result):
                    return result
    
    # Fallback: greedy
    values = [primes[0]]
    for p in primes[1:]:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    
    return values

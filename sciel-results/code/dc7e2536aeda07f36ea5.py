"""Rewritten program: Longest non-decreasing-gap prime subsequence via DAG longest path."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Sieve of Eratosthenes for primes <= 10000
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0

    primes = np.array([p for p in range(2, limit + 1) if is_prime[p]], dtype=np.int32)
    n = len(primes)

    if n == 0:
        return []

    # DP arrays for longest path in DAG:
    # dp[i] = length of longest valid subsequence ending at primes[i]
    # last_gap[i] = the gap from predecessor to primes[i] (0 if no predecessor)
    # pred[i] = index of predecessor in optimal path ending at i
    dp = np.ones(n, dtype=np.int32)
    last_gap = np.zeros(n, dtype=np.int32)
    pred = np.full(n, -1, dtype=np.int32)

    # Process each node i; for each i, find best predecessor j < i
    # where gap = primes[i] - primes[j] >= last_gap[j]
    # among valid j, maximize dp[j], tie-break by minimizing new gap (max primes[j])
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        pi = primes[i]
        gaps = pi - primes[:i]  # shape (i,)
        
        # Valid if gap >= last_gap[j]
        valid_mask = gaps >= last_gap[:i]
        
        if not np.any(valid_mask):
            continue
        
        valid_indices = np.where(valid_mask)[0]
        lens = dp[valid_indices]
        max_len_val = np.max(lens)
        
        # Among those with max length, pick the one with largest prime (smallest gap)
        cand_mask = lens == max_len_val
        candidates = valid_indices[cand_mask]
        
        best_j = candidates[np.argmax(primes[candidates])]
        
        dp[i] = max_len_val + 1
        last_gap[i] = pi - primes[best_j]
        pred[i] = best_j

    # Find global maximum
    max_len_global = np.max(dp)
    best_indices = np.where(dp == max_len_global)[0]
    best_idx = int(best_indices[0])

    # Reconstruct
    seq = []
    cur = best_idx
    while cur != -1:
        seq.append(int(primes[cur]))
        cur = int(pred[cur])
    seq.reverse()

    return seq
# EVOLVE-BLOCK-END

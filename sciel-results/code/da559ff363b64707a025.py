# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    from math import isqrt
    
    # 1. Generate primes up to 10000 using Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # 2. DP: dp[i] is a list of (last_gap, length, prev_idx) for Pareto-optimal entries
    #    We maintain Pareto optimality: no entry dominates another.
    #    Entry A dominates B if A.last_gap <= B.last_gap and A.length >= B.length
    
    dp = [[] for _ in range(N)]
    
    # Base case: each prime alone has last_gap=0, length=1, no prev
    for i in range(N):
        dp[i].append((0, 1, -1))
    
    # Fill DP
    for i in range(1, N):
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            # From dp[j], find the entry with max length where last_gap <= gap
            best_len = 0
            for (lg, ln, pv) in dp[j]:
                if lg <= gap:
                    if ln > best_len:
                        best_len = ln
            if best_len > 0:
                new_len = best_len + 1
                new_entry = (gap, new_len, j)
                # Check if this entry is dominated by existing entries in dp[i]
                dominated = False
                for (eg, el, ep) in dp[i]:
                    if eg <= gap and el >= new_len:
                        dominated = True
                        break
                if not dominated:
                    # Remove entries in dp[i] dominated by new_entry
                    new_dp_i = []
                    for (eg, el, ep) in dp[i]:
                        if not (gap <= eg and new_len >= el):
                            new_dp_i.append((eg, el, ep))
                    new_dp_i.append(new_entry)
                    dp[i] = new_dp_i
        
        # Time check
        if time.time() - start_time > DEADLINE:
            break
    
    # 3. Find the best entry across all dp[i]
    max_len = 0
    best_i = 0
    best_prev = -1
    for i in range(N):
        for (lg, ln, pv) in dp[i]:
            if ln > max_len:
                max_len = ln
                best_i = i
                best_prev = pv
    
    # 4. Reconstruct the sequence
    if max_len <= 1:
        return [primes[0]]
    
    seq = [primes[best_i]]
    cur = best_i
    prev_idx = best_prev
    while prev_idx != -1:
        seq.append(primes[prev_idx])
        gap_to_cur = primes[cur] - primes[prev_idx]
        # Find entry in dp[prev_idx] with length = len(seq) and last_gap <= gap_to_cur
        target_len = len(seq)
        found = False
        for (lg, ln, pv2) in dp[prev_idx]:
            if ln == target_len and lg <= gap_to_cur:
                cur = prev_idx
                prev_idx = pv2
                found = True
                break
        if not found:
            # Fallback
            for (lg, ln, pv2) in dp[prev_idx]:
                if ln == target_len:
                    cur = prev_idx
                    prev_idx = pv2
                    found = True
                    break
        if not found:
            break
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END

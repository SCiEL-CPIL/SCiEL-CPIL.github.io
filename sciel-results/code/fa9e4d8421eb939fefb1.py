"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    
    # 1. Generate primes up to 10000
    is_prime = [True] * 10001
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(10000) + 1):
        if is_prime[i]:
            for j in range(i*i, 10001, i):
                is_prime[j] = False
                
    primes = [p for p in range(2, 10001) if is_prime[p]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
        
    # 2. DP setup
    # dp[i] is a list of tuples (gap, length)
    # parent[i] is a list of tuples (prev_idx, prev_idx_in_dp)
    dp = [[] for _ in range(N)]
    parent = [[] for _ in range(N)]
    
    # Base case: sequence of length 1
    # We treat the "gap" of a single element as 0
    dp[0] = [(0, 1)]
    parent[0] = [(-1, -1)]
    
    best_len = 1
    best_i = 0
    best_k = 0
    
    # 3. Fill DP table
    for i in range(1, N):
        # candidates: gap -> (length, prev_idx, prev_idx_in_dp)
        candidates = {}
        
        for j in range(i):
            g = primes[i] - primes[j]
            # Iterate through all valid states ending at j
            for k in range(len(dp[j])):
                g_prev, L_prev = dp[j][k]
                if g_prev <= g:
                    L_new = L_prev + 1
                    # Check if this gap g is already in candidates
                    if g in candidates:
                        # Keep the one with larger length
                        if candidates[g][0] < L_new:
                            candidates[g] = (L_new, j, k)
                    else:
                        candidates[g] = (L_new, j, k)
        
        # Prune candidates to keep Pareto frontier
        # Sort by gap
        if not candidates:
            continue
            
        sorted_gaps = sorted(candidates.keys())
        
        current_dp_i = []
        current_parent_i = []
        max_len_so_far = 0
        
        for g in sorted_gaps:
            L, j, k = candidates[g]
            if L > max_len_so_far:
                current_dp_i.append((g, L))
                current_parent_i.append((j, k))
                max_len_so_far = L
                # Update global best
                if L > best_len:
                    best_len = L
                    best_i = i
                    best_k = len(current_dp_i) - 1
        
        dp[i] = current_dp_i
        parent[i] = current_parent_i

    # 4. Reconstruct sequence
    if best_len == 1:
        return [primes[best_i]]
        
    seq = []
    curr_i = best_i
    curr_k = best_k
    
    while curr_i != -1:
        seq.append(primes[curr_i])
        if curr_i == 0:
            break
        prev_i, prev_k = parent[curr_i][curr_k]
        curr_i = prev_i
        curr_k = prev_k
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END

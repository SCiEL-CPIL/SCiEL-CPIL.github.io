def construct():
    import time
    import bisect
    
    start_time = time.time()
    DEADLINE = 80.0

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = 0
        return [i for i in range(n+1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    # dp[i] is a list of (gap, length) tuples, sorted by gap.
    # Lengths are strictly increasing.
    dp = [None] * n
    parent = [None] * n # parent[i] will be a list of (prev_idx, prev_gap) corresponding to dp[i]
    
    best_length = 1
    best_i = 0
    best_gap = 0
    
    for i in range(n):
        # Initialize frontier for i
        # State: first element, gap=0, length=1
        current_frontier = [(0, 1)]
        current_parents = [None]
        
        for j in range(i):
            if dp[j] is None:
                continue
                
            gap = primes[i] - primes[j]
            frontier_j = dp[j]
            
            # Find rightmost index where gap_j <= gap
            idx = bisect.bisect_right(frontier_j, (gap, float('inf')))
            
            if idx > 0:
                prev_gap, prev_len = frontier_j[idx - 1]
                new_len = prev_len + 1
                
                # Check if this new state (gap, new_len) is dominated by current_frontier
                ins_idx = bisect.bisect_right(current_frontier, (gap, float('inf')))
                
                # Check if dominated by state at ins_idx - 1
                if ins_idx > 0:
                    existing_gap, existing_len = current_frontier[ins_idx - 1]
                    if existing_len >= new_len:
                        # Dominated
                        continue
                
                # Check if it dominates states at ins_idx and onwards
                end_idx = ins_idx
                while end_idx < len(current_frontier):
                    g_e, l_e = current_frontier[end_idx]
                    if l_e <= new_len:
                        end_idx += 1
                    else:
                        break
                
                # Replace slice [ins_idx, end_idx) with new state
                current_frontier[ins_idx:end_idx] = [(gap, new_len)]
                current_parents[ins_idx:end_idx] = [(j, prev_gap)]
                
                if new_len > best_length:
                    best_length = new_len
                    best_i = i
                    best_gap = gap
        
        dp[i] = current_frontier
        parent[i] = current_parents
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Reconstruct
    if best_length <= 1:
        return [primes[0]]
    
    seq = []
    i = best_i
    gap = best_gap
    
    # Find the index of (gap, best_length) in dp[i]
    target_idx = -1
    for k, (g, l) in enumerate(dp[i]):
        if g == gap and l == best_length:
            target_idx = k
            break
    
    if target_idx == -1:
        return [primes[0]]
    
    # Walk back
    while True:
        seq.append(primes[i])
        p = parent[i][target_idx]
        if p is None:
            break
        j, prev_gap = p
        i = j
        gap = prev_gap
        
        # Find index in dp[i] for (gap, length-1)
        target_idx = -1
        for k, (g, l) in enumerate(dp[i]):
            if g == gap:
                target_idx = k
                break
        
        if target_idx == -1:
            break
    
    seq.reverse()
    return seq

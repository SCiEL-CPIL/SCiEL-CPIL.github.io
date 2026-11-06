import time

def construct():
    t0 = time.time()
    DEADLINE = 50.0  # Safety margin

    N = 10000
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)

    # dp[i] is a list of (gap, length) tuples, sorted by gap.
    # Lengths are strictly increasing.
    dp = [[] for _ in range(n)]

    # Helper to reduce a list of (gap, length) candidates to Pareto frontier
    def reduce_frontier(cands):
        if not cands:
            return []
        # Dictionary to keep max length for each gap
        best_by_gap = {}
        for g, l in cands:
            if g not in best_by_gap or l > best_by_gap[g]:
                best_by_gap[g] = l
        
        # Sort by gap
        sorted_items = sorted(best_by_gap.items())
        
        frontier = []
        max_len_so_far = 0
        for g, l in sorted_items:
            if l > max_len_so_far:
                frontier.append((g, l))
                max_len_so_far = l
            # else dominated
        return frontier

    for i in range(n):
        if time.time() - t0 > DEADLINE:
            break
            
        pi = primes[i]
        cands = []
        
        for j in range(i):
            gap = pi - primes[j]
            
            # Option 1: Start new sequence
            cands.append((gap, 2))
            
            # Option 2: Extend existing sequence from j
            if dp[j]:
                # Binary search in dp[j] for largest gap <= current gap
                # dp[j] is sorted by gap
                lo, hi = 0, len(dp[j]) - 1
                best_k = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if dp[j][mid][0] <= gap:
                        best_k = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                
                if best_k != -1:
                    prev_gap, prev_len = dp[j][best_k]
                    cands.append((gap, prev_len + 1))
        
        dp[i] = reduce_frontier(cands)

    # Find the best sequence
    best_len = 1
    best_i = 0
    best_k = -1
    
    for i in range(n):
        for k, (g, l) in enumerate(dp[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_k = k

    # Reconstruct
    if best_len <= 1:
        return [primes[0]]

    result = []
    cur_i = best_i
    cur_k = best_k
    
    # We need to trace back.
    # The state (gap, length) at dp[i][k] comes from some j and some state in dp[j].
    # But we didn't store the parent pointers in the current simplified dp structure.
    # We need to store parent pointers to reconstruct.
    
    # Let's modify dp to store (gap, length, prev_idx, prev_entry_idx)
    
    # Re-run with parent pointers
    dp = [[] for _ in range(n)]
    
    def reduce_frontier_with_parents(cands):
        if not cands:
            return []
        best_by_gap = {}
        for g, l, pi, pei in cands:
            if g not in best_by_gap or l > best_by_gap[g][0]:
                best_by_gap[g] = (l, pi, pei)
        
        sorted_items = sorted(best_by_gap.items())
        
        frontier = []
        max_len_so_far = 0
        for g, (l, pi, pei) in sorted_items:
            if l > max_len_so_far:
                frontier.append((g, l, pi, pei))
                max_len_so_far = l
        return frontier

    for i in range(n):
        if time.time() - t0 > DEADLINE:
            break
        pi = primes[i]
        cands = []
        
        for j in range(i):
            gap = pi - primes[j]
            
            # Option 1: Start new
            cands.append((gap, 2, j, -1))
            
            # Option 2: Extend
            if dp[j]:
                lo, hi = 0, len(dp[j]) - 1
                best_k = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if dp[j][mid][0] <= gap:
                        best_k = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                
                if best_k != -1:
                    prev_gap, prev_len, prev_pi, prev_pei = dp[j][best_k]
                    cands.append((gap, prev_len + 1, j, best_k))
        
        dp[i] = reduce_frontier_with_parents(cands)

    # Find best
    best_len = 1
    best_i = 0
    best_k = -1
    
    for i in range(n):
        for k, (g, l, pi, pei) in enumerate(dp[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_k = k

    # Reconstruct
    if best_len <= 1:
        return [primes[0]]

    result = []
    cur_i = best_i
    cur_k = best_k
    
    while True:
        result.append(primes[cur_i])
        if cur_k == -1:
            break
        g, l, pi, pei = dp[cur_i][cur_k]
        cur_i = pi
        cur_k = pei
    
    result.reverse()
    return result

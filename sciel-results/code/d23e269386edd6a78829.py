import time
import bisect

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

    # dp[i] is a list of (gap, length, prev_idx, prev_entry_idx) tuples, sorted by gap.
    # Lengths are strictly increasing.
    dp = [[] for _ in range(n)]
    
    # Pre-extract gaps as separate lists for faster binary search
    dp_gaps = [None] * n  # dp_gaps[i] = list of gaps in dp[i]
    dp_lens = [None] * n  # dp_lens[i] = list of lengths in dp[i]

    def reduce_frontier_with_parents(cands):
        if not cands:
            return [], [], []
        # Dictionary to keep max length for each gap
        best_by_gap = {}
        for g, l, pi, pei in cands:
            if g not in best_by_gap or l > best_by_gap[g][0]:
                best_by_gap[g] = (l, pi, pei)
        
        sorted_items = sorted(best_by_gap.items())
        
        frontier = []
        gaps_list = []
        lens_list = []
        max_len_so_far = 0
        for g, (l, pi, pei) in sorted_items:
            if l > max_len_so_far:
                frontier.append((g, l, pi, pei))
                gaps_list.append(g)
                lens_list.append(l)
                max_len_so_far = l
        return frontier, gaps_list, lens_list

    for i in range(n):
        if time.time() - t0 > DEADLINE:
            break
            
        pi = primes[i]
        cands = []
        
        for j in range(i):
            gap = pi - primes[j]
            
            # Option 1: Start new sequence
            cands.append((gap, 2, j, -1))
            
            # Option 2: Extend existing sequence from j
            if dp[j]:
                # Binary search in dp[j] for largest gap <= current gap
                jgaps = dp_gaps[j]
                idx = bisect.bisect_right(jgaps, gap) - 1
                if idx >= 0:
                    prev_len = dp_lens[j][idx]
                    cands.append((gap, prev_len + 1, j, idx))
        
        frontier, gaps_list, lens_list = reduce_frontier_with_parents(cands)
        dp[i] = frontier
        dp_gaps[i] = gaps_list
        dp_lens[i] = lens_list

    # Find the best sequence
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

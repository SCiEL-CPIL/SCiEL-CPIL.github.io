import math

def construct():
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    # dp[i] is a list of (gap, length)
    # par[i] is a dict: gap -> (prev_idx, prev_gap)
    dp = [None] * n
    par = [None] * n
    
    global_max_len = 0
    global_best_i = 0
    global_best_gap = 0
    
    for i in range(n):
        p_i = primes[i]
        candidates = [(0, 1, -1, -1)] # (gap, len, prev_idx, prev_gap)
        
        for j in range(i):
            p_j = primes[j]
            gap_ij = p_i - p_j
            
            # dp[j] is sorted by gap ascending
            for gap_j, len_j in dp[j]:
                if gap_j > gap_ij:
                    break
                new_gap = gap_ij
                new_len = len_j + 1
                candidates.append((new_gap, new_len, j, gap_j))
                
        # Process candidates to form Pareto frontier for i
        # Sort by gap, then by length descending (to handle ties in gap, keep max len)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        cur_dp = []
        cur_par = {}
        max_len_so_far = 0
        
        for gap, length, prev_idx, prev_gap in candidates:
            if length > max_len_so_far:
                cur_dp.append((gap, length))
                cur_par[gap] = (prev_idx, prev_gap)
                max_len_so_far = length
                
        dp[i] = cur_dp
        par[i] = cur_par
        
        if max_len_so_far > global_max_len:
            global_max_len = max_len_so_far
            global_best_i = i
            global_best_gap = max_len_so_far # Wait, we need the gap associated with the max length
            # Actually, the last item in cur_dp has the max length.
            # But there might be multiple with same max length? No, length is strictly increasing in cur_dp.
            # So the last item is the unique max length for this i.
            # But we want the global max.
            # If max_len_so_far is the max for i, we store it.
            # We need to store the gap to reconstruct.
            # The gap is cur_dp[-1][0].
            global_best_gap = cur_dp[-1][0]
            
    # Reconstruct sequence
    seq = []
    i = global_best_i
    gap = global_best_gap
    while i != -1:
        seq.append(primes[i])
        if gap == -1:
            break
        prev_idx, prev_gap = par[i][gap]
        i = prev_idx
        gap = prev_gap
        
    seq.reverse()
    return seq

# Test
# print(construct())

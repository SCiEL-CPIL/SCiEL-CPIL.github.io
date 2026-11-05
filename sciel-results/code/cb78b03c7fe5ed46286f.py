import time

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin for 90s limit
    
    # 1. Sieve of Eratosthenes for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    M = len(primes)
    
    # Map prime value to index for O(1) lookup if needed, but we iterate indices directly
    # We need to access best[j][g] where g = primes[i] - primes[j]
    
    # Max possible gap is 10000
    G = limit
    
    # dp[i][g]: length of longest seq ending at primes[i] with last gap g
    # best[i][g]: max of dp[i][0..g]
    # prev[i][g]: index j that gave the max for dp[i][g]
    
    # Using lists of lists. 
    # M ~ 1229, G ~ 10000.
    # 1229 * 10001 * 3 lists of ints.
    # This is about 36 million ints.
    # Python int is 28 bytes. 36M * 28 ~ 1GB. Too much.
    
    # We must use array('i') or numpy.
    # Let's use array('i') from array module.
    from array import array
    
    # Initialize dp, best, prev
    # dp[i] will be computed on the fly and stored? 
    # We need best[j] for all j < i. So we must store best.
    # We need prev[i] for backtracking. So we must store prev.
    # We don't need to store dp[i] after best[i] is computed, 
    # but we need it to compute best[i].
    
    best = [array('i', [0]) * (G + 1) for _ in range(M)]
    prev = [array('i', [-1]) * (G + 1) for _ in range(M)]
    
    # We don't need to store all dp rows, just the current one to compute best row.
    # But wait, best[i] depends on dp[i].
    # We can compute dp_i as a local array, then compute best_i, then store best_i.
    
    max_len = 0
    max_i = -1
    max_g = -1
    
    for i in range(M):
        if time.time() > deadline - 5:
            break
            
        p_i = primes[i]
        # dp_i[g] for g in 0..G
        dp_i = array('i', [0]) * (G + 1)
        
        # Base case: sequence of length 1
        # This corresponds to "last gap" 0.
        # Any gap g >= 0 is valid as a next gap.
        dp_i[0] = 1
        
        # Transitions from j < i
        for j in range(i):
            p_j = primes[j]
            g = p_i - p_j
            # g is always <= G
            # We want to update dp_i[g]
            # Candidate length is 1 + best[j][g]
            # best[j] is already computed
            val = 1 + best[j][g]
            if val > dp_i[g]:
                dp_i[g] = val
                prev[i][g] = j
        
        # Compute best_i: prefix max of dp_i
        best_i = array('i', [0]) * (G + 1)
        current_max = 0
        for g in range(G + 1):
            if dp_i[g] > current_max:
                current_max = dp_i[g]
            best_i[g] = current_max
            if current_max > max_len:
                max_len = current_max
                max_i = i
                max_g = g
        
        best[i] = best_i
        
    if max_len == 0:
        return [2] # Fallback
        
    # Backtrack
    path = []
    i = max_i
    g = max_g
    while i != -1:
        path.append(primes[i])
        j = prev[i][g]
        if j == -1:
            break
        # The gap used to get to i was g.
        # The previous state was at j.
        # What was the gap at j?
        # We don't store the gap explicitly in the path, but we know the transition was from j to i with gap g.
        # So the state at j had some gap g_prev <= g.
        # Actually, we need to find the g_prev that was used.
        # The value dp[i][g] came from best[j][g].
        # best[j][g] is the max of dp[j][0..g].
        # It's possible that the max was achieved at some g' < g.
        # We need to know which g' was used to correctly backtrack.
        # 
        # Problem: prev[i][g] stores j, but not the g_prev.
        # If best[j][g] came from dp[j][g'] where g' < g, we need to know g'.
        # 
        # Solution: Store prev_gap[i][g] as well?
        # Or, since we iterate g in increasing order in the prefix max, 
        # we can store the index of the max?
        # 
        # Let's modify the DP to store the previous gap index.
        # prev_gap[i][g] = the g' in [0..g] that achieved best[j][g].
        # But best[j][g] is precomputed.
        # 
        # Alternative:
        # When updating dp_i[g] from j, we know the candidate is 1 + best[j][g].
        # But best[j][g] might come from any g' <= g.
        # To backtrack, we need to know which g' was chosen.
        # 
        # We can store `prev_gap[i][g]` which is the gap value $g'$ at node $j$ that was used.
        # How to find it?
        # best[j][g] is the max of dp[j][0..g].
        # We can find the largest g' <= g such that dp[j][g'] == best[j][g]?
        # Not necessarily unique. But any one works.
        # 
        # To avoid storing extra arrays, we can just store the previous index `j` and the previous gap `g_prev`.
        # We need to know `g_prev`.
        # 
        # Let's add `prev_gap` array.
        
    # Re-run with prev_gap storage
    # This requires re-doing the DP loop or modifying the previous one.
    # Since the code is not executed yet, I will write the final version with `prev_gap`.
    
    return path

# Revised Code Structure

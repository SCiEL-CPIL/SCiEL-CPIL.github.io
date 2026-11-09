import bisect

def construct():
    # Generate primes up to 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(10000)
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [primes[0]]

    # dp[i][j] = length of longest valid sequence ending with primes[j], primes[i] (j < i)
    # par[i][j] = index k such that the sequence ending with primes[j], primes[i] 
    #             was extended from a sequence ending with primes[k], primes[j]
    
    dp = [[0] * N for _ in range(N)]
    par = [[-1] * N for _ in range(N)]
    
    # Suf[i] will hold suffix maximums for dp[i][0..i-1]
    # Suf[i][k] = max(dp[i][k], dp[i][k+1], ..., dp[i][i-1])
    Suf = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = -1 # -1 indicates length 1
    
    # We iterate i from 0 to N-1
    # For a fixed i, we compute dp[i][j] for j < i
    # These computations depend on Suf[j]
    # Suf[j] must be ready.
    # Suf[j] is computed after we finish processing row j (i.e., after inner loop of i=j).
    
    for i in range(N):
        # Compute dp[i][j] for j from i-1 down to 0
        for j in range(i - 1, -1, -1):
            gap = primes[i] - primes[j]
            limit = primes[j] - gap
            
            # Find smallest k such that primes[k] >= limit
            k_min = bisect.bisect_left(primes, limit)
            
            if k_min < j:
                # Valid k exists in [k_min, j-1]
                # We need max dp[j][k] for k in [k_min, j-1]
                # This is Suf[j][k_min]
                # Suf[j] should be available since j < i
                if Suf[j] is not None:
                    best_val = Suf[j][k_min]
                    if best_val > 0:
                        dp[i][j] = best_val + 1
                        # Find which k gave this best_val
                        # We need to find k in [k_min, j-1] such that dp[j][k] == best_val
                        # Since Suf[j][k] is max from k to end, 
                        # we can just scan backwards from j-1 to k_min to find the first k with dp[j][k] == best_val
                        # Or we can store argmax in Suf? 
                        # Storing argmax is better.
                        # Let's redefine Suf to store (value, index)
                        pass
                else:
                    # This case should not happen if logic is correct, 
                    # because Suf[j] is computed after i=j loop finishes.
                    # Since i > j, i=j loop is already done.
                    dp[i][j] = 2
            else:
                dp[i][j] = 2
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
        
        # After processing all j for this i, compute Suf[i]
        # Suf[i][k] = max(dp[i][k], Suf[i][k+1])
        # We also need to know which index gave the max for backtracking.
        # Let's use a list of tuples (val, idx) for Suf.
        
    # The above logic for backtracking is slightly messy with just Suf values.
    # Let's refine the Suf structure to store the index of the max.
    
    # Re-initialize
    dp = [[0] * N for _ in range(N)]
    # Suf[i] is a list of tuples (max_val, idx) for suffixes of dp[i]
    Suf = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = -1
    
    for i in range(N):
        for j in range(i - 1, -1, -1):
            gap = primes[i] - primes[j]
            limit = primes[j] - gap
            k_min = bisect.bisect_left(primes, limit)
            
            if k_min < j:
                if Suf[j] is not None:
                    # Suf[j][k_min] is (max_val, idx)
                    val, idx = Suf[j][k_min]
                    if val > 0:
                        dp[i][j] = val + 1
                        # The predecessor of (i,j) is (j, idx)
                        # We need to store this link.
                        # Let's use a separate par array.
                        pass
                else:
                    dp[i][j] = 2
            else:
                dp[i][j] = 2
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
        
        # Compute Suf[i]
        # dp[i][0..i-1]
        # Suf[i][k] = max(dp[i][k..i-1])
        # We store (value, index)
        suf_list = [None] * i
        if i > 0:
            max_val = dp[i][i-1]
            max_idx = i-1
            suf_list[i-1] = (max_val, max_idx)
            for k in range(i-2, -1, -1):
                if dp[i][k] > max_val:
                    max_val = dp[i][k]
                    max_idx = k
                suf_list[k] = (max_val, max_idx)
            Suf[i] = suf_list
        else:
            Suf[i] = []

    # Now we need to reconstruct the path.
    # We need par[i][j] to know which k was chosen.
    # In the loop above, we didn't store par.
    # We can re-run the DP to store par, or store it in the first pass.
    # Let's combine them.
    
    # Final Implementation Plan:
    # 1. Sieve
    # 2. DP with par
    # 3. Backtrack
    # 4. Return list

    # Re-doing with par
    dp = [[0] * N for _ in range(N)]
    par = [[-1] * N for _ in range(N)]
    Suf = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = -1
    
    for i in range(N):
        for j in range(i - 1, -1, -1):
            gap = primes[i] - primes[j]
            limit = primes[j] - gap
            k_min = bisect.bisect_left(primes, limit)
            
            if k_min < j:
                if Suf[j] is not None and len(Suf[j]) > 0:
                    val, idx = Suf[j][k_min]
                    if val > 0:
                        dp[i][j] = val + 1
                        par[i][j] = idx
                        if dp[i][j] > max_len:
                            max_len = dp[i][j]
                            best_i = i
                            best_j = j
                        continue
            # If no valid k or val was 0, it's a base pair
            dp[i][j] = 2
            par[i][j] = -1
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
        
        # Compute Suf[i]
        if i > 0:
            suf_list = [None] * i
            max_val = dp[i][i-1]
            max_idx = i-1
            suf_list[i-1] = (max_val, max_idx)
            for k in range(i-2, -1, -1):
                if dp[i][k] > max_val:
                    max_val = dp[i][k]
                    max_idx = k
                suf_list[k] = (max_val, max_idx)
            Suf[i] = suf_list
        else:
            Suf[i] = []

    # Backtrack
    if max_len == 1:
        return [primes[0]]
    
    seq = []
    i = best_i
    j = best_j
    while j != -1:
        seq.append(primes[i])
        i = j
        j = par[i][j]
    seq.append(primes[i])
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END

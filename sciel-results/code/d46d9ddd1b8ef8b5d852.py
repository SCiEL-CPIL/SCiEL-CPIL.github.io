def construct():
    import time
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    # Sieve for primes up to 10000
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
    prime_set = set(primes)
    
    # Map prime value to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # dp[i] = dict mapping gap -> max length ending at primes[i] with last gap = gap
    # Also maintain prefix max for efficient queries
    # dp[i][g] = max length of valid subsequence ending at primes[i] with last gap exactly g
    # We want: for each i, and for each possible previous prime j < i,
    #   gap = primes[i] - primes[j]
    #   dp[i][gap] = max(dp[i][gap], 1 + max_{g' <= gap} dp[j][g'])
    
    # To make this efficient, for each j, we'll maintain a sorted list of (gap, value) 
    # and a prefix max array.
    
    # dp[i] will be a dict: gap -> best length
    # prefix_max[i] will be a list of (gap_threshold, cumulative_max) for O(log) or O(1) lookup
    
    # Actually, let's store for each i:
    #   gaps_i = sorted list of distinct gaps used
    #   vals_i = corresponding dp values
    #   prefix_vals_i = prefix max of vals_i
    
    dp = [{} for _ in range(n)]  # dp[i][gap] = max length
    prefix_data = [None] * n  # will store (sorted_gaps, prefix_max_vals)
    
    # Initialize: every prime alone has length 1
    # We'll build up from left to right
    
    best_length = 1
    best_seq = [primes[0]]
    
    for i in range(n):
        # First, set base case: length 1
        if not dp[i]:
            dp[i][0] = 1  # gap 0 means "first element"
            # Actually, let's use gap=0 as sentinel for "no previous gap"
        
        for j in range(i):
            gap = primes[i] - primes[j]
            # Find max_{g' <= gap} dp[j][g']
            # Use prefix_data[j]
            if prefix_data[j] is not None:
                sorted_gaps, prefix_max_vals = prefix_data[j]
                # Find rightmost index where sorted_gaps[idx] <= gap
                # Use binary search
                import bisect
                idx = bisect.bisect_right(sorted_gaps, gap)
                if idx > 0:
                    max_prev = prefix_max_vals[idx - 1]
                    new_len = 1 + max_prev
                    if new_len > dp[i].get(gap, 0):
                        dp[i][gap] = new_len
                        if new_len > best_length:
                            best_length = new_len
            else:
                # j has no prefix data yet (shouldn't happen if we process in order)
                pass
        
        # Build prefix data for i
        if dp[i]:
            items = sorted(dp[i].items())  # list of (gap, val)
            sorted_gaps = [g for g, v in items]
            vals = [v for g, v in items]
            prefix_max_vals = [0] * len(vals)
            prefix_max_vals[0] = vals[0]
            for k in range(1, len(vals)):
                prefix_max_vals[k] = max(prefix_max_vals[k-1], vals[k])
            prefix_data[i] = (sorted_gaps, prefix_max_vals)
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    # Now reconstruct the best sequence
    # We need to track parent pointers. Let me redo with parent tracking.
    
    # Actually, let's restart with parent tracking for reconstruction
    start_time = time.time()
    
    dp = [{} for _ in range(n)]
    parent = [{} for _ in range(n)]  # parent[i][gap] = (j, prev_gap) that gave best
    prefix_data = [None] * n
    
    best_length = 1
    best_i = 0
    best_gap = 0
    
    for i in range(n):
        dp[i][0] = 1
        parent[i][0] = None
        
        for j in range(i):
            gap = primes[i] - primes[j]
            if prefix_data[j] is not None:
                sorted_gaps, prefix_max_vals = prefix_data[j]
                import bisect
                idx = bisect.bisect_right(sorted_gaps, gap)
                if idx > 0:
                    max_prev = prefix_max_vals[idx - 1]
                    new_len = 1 + max_prev
                    if new_len > dp[i].get(gap, 0):
                        dp[i][gap] = new_len
                        # Find which g' gave the max
                        # We need to find the actual j and g' that achieved max_prev
                        # This is tricky. Let me store more info.
                        parent[i][gap] = (j, None)  # need to fix
                        if new_len > best_length:
                            best_length = new_len
                            best_i = i
                            best_gap = gap
        
        if dp[i]:
            items = sorted(dp[i].items())
            sorted_gaps = [g for g, v in items]
            vals = [v for g, v in items]
            prefix_max_vals = [0] * len(vals)
            prefix_max_vals[0] = vals[0]
            for k in range(1, len(vals)):
                prefix_max_vals[k] = max(prefix_max_vals[k-1], vals[k])
            prefix_data[i] = (sorted_gaps, prefix_max_vals)
        
        if time.time() - start_time > DEADLINE:
            break
    
    # The reconstruction is complex. Let me use a simpler approach:
    # Just track the actual sequence during DP by storing the full path, 
    # or use a different method.
    
    # Alternative: since n is only 1229, let me just do a cleaner DP with 
    # explicit parent tracking.
    
    # Let me restart completely with a cleaner approach.
    start_time = time.time()
    
    # dp[i][j] = max length of valid subsequence ending at primes[i] with previous prime at index j
    # This is O(n^2) states, O(n) transitions each = O(n^3) total
    # n=1229, n^3 = 1.85e9 - too slow
    
    # Better: dp[i][g] approach with proper parent tracking
    # For reconstruction, when we update dp[i][gap] from dp[j][g'], we need to know which g' was used
    # Let's store: best_prev[j][gap] = (best_length, best_g_prime) for prefix max queries
    
    # Actually, let me just store the full sequence for the best answer found.
    # Given the small size, I'll use a greedy + local search hybrid.
    
    # Let me try a different approach: iterate over possible starting points and use greedy
    # with backtracking for small cases.
    
    # Actually, let me just do the DP properly and reconstruct.
    
    # dp[i][g] = max length, and we store parent[i][g] = (j, g_prime) 
    # where g_prime is the specific gap at j that achieved the prefix max
    
    # To handle the prefix max correctly, I need to store not just the max value but
    # also which gap achieved it.
    
    start_time = time.time()
    
    # For each i, maintain:
    #   sorted list of (gap, value, best_gap_for_prefix) 
    # where best_gap_for_prefix[k] = the gap index that achieves the prefix max up to k
    
    dp = [{} for _ in range(n)]
    parent = [{} for _ in range(n)]
    
    # prefix_info[i] = (sorted_gaps, prefix_max_vals, prefix_best_gap_idx)
    # prefix_best_gap_idx[k] = index in sorted_gaps that gives the max value in prefix_max_vals[k]
    
    prefix_info = [None] * n
    
    best_length = 1
    best_end = (0, 0)  # (i, gap)
    
    for i in range(n):
        dp[i][0] = 1
        parent[i][0] = None
        
        for j in range(i):
            gap = primes[i] - primes[j]
            if prefix_info[j] is not None:
                sorted_gaps, prefix_max_vals, prefix_best_idx = prefix_info[j]
                import bisect
                idx = bisect.bisect_right(sorted_gaps, gap)
                if idx > 0:
                    max_prev = prefix_max_vals[idx - 1]
                    best_prev_gap_idx = prefix_best_idx[idx - 1]
                    best_prev_gap = sorted_gaps[best_prev_gap_idx]
                    new_len = 1 + max_prev
                    if new_len > dp[i].get(gap, 0):
                        dp[i][gap] = new_len
                        parent[i][gap] = (j, best_prev_gap)
                        if new_len > best_length:
                            best_length = new_len
                            best_end = (i, gap)
        
        if dp[i]:
            items = sorted(dp[i].items())
            sorted_gaps = [g for g, v in items]
            vals = [v for g, v in items]
            prefix_max_vals = [0] * len(vals)
            prefix_best_idx = [0] * len(vals)
            prefix_max_vals[0] = vals[0]
            prefix_best_idx[0] = 0
            for k in range(1, len(vals)):
                if vals[k] > prefix_max_vals[k-1]:
                    prefix_max_vals[k] = vals[k]
                    prefix_best_idx[k] = k
                else:
                    prefix_max_vals[k] = prefix_max_vals[k-1]
                    prefix_best_idx[k] = prefix_best_idx[k-1]
            prefix_info[i] = (sorted_gaps, prefix_max_vals, prefix_best_idx)
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Reconstruct sequence
    if best_length <= 1:
        return [primes[0]]
    
    seq = []
    i, gap = best_end
    while i is not None:
        seq.append(primes[i])
        p = parent[i].get(gap)
        if p is None:
            break
        j, prev_gap = p
        i, gap = j, prev_gap
    
    seq.reverse()
    return seq

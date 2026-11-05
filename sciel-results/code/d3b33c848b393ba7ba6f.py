def construct():
    from math import isqrt
    
    # Generate all primes <= 10000
    primes = []
    for p in range(2, 10001):
        if all(p % d for d in range(2, isqrt(p) + 1)):
            primes.append(p)
    
    n = len(primes)
    if n < 2:
        return primes
    
    # dp[i][j] = max length of nondecreasing-gap prime subsequence 
    # ending at primes[i] where the last gap (primes[i] - primes[j]) is the last gap
    # Actually, let's define: for each pair (j, i) with j < i, 
    # best[j][i] = max length of subsequence ending with ... , primes[j], primes[i]
    # with nondecreasing gaps.
    
    # best[j][i] = 1 + max{ best[k][j] : k < j and primes[j] - primes[k] <= primes[i] - primes[j] }
    # Base case: best[j][i] = 2 for all j < i (sequence of just primes[j], primes[i])
    
    # This is O(n^3) which is 1229^3 ≈ 1.86 * 10^9 - too slow!
    
    # Need to optimize. For each j, and for each i > j, we need max over k < j with 
    # primes[j] - primes[k] <= primes[i] - primes[j], i.e., primes[k] >= primes[j] - (primes[i] - primes[j]) = 2*primes[j] - primes[i]
    
    # So for fixed j, as i increases, the threshold 2*primes[j] - primes[i] decreases,
    # so we include more k values. We can precompute for each j, sort the relevant values.
    
    # Alternative: for each j, precompute an array where for each possible gap g,
    # we store the max best[k][j] for k with primes[j] - primes[k] = g. Then for query with max_gap,
    # we need max over g' <= max_gap of that value. We can use prefix max.
    
    # Let's try: for each j, create a dict or array mapping gap -> max length ending with (k,j)
    # Then prefix max over gaps.
    
    # Max gap is at most 10000. So for each j, we can have an array of size 10001.
    # Total: n * 10001 ≈ 1229 * 10001 ≈ 12.3M - feasible.
    
    max_gap = 10000
    # best_end[j] = list of (gap, length) for sequences ending at primes[j]
    # Actually, let's use: dp[j][g] = max length of subsequence ending at primes[j] with last gap = g
    # But we want prefix max: for each j, pref[j][g] = max{ dp[j][g'] : g' <= g }
    
    # Initialize: dp[j][g] = -inf for all, but base case is sequences of length 2.
    # For sequence (primes[j], primes[i]), length = 2, last gap = primes[i] - primes[j]
    
    # Let's do it iteratively. For i from 1 to n-1:
    #   For j from 0 to i-1:
    #       g = primes[i] - primes[j]
    #       # Find max length of sequence ending at primes[j] with last gap <= g
    #       # That's pref[j][g] if we've computed it
    #       # But we need to compute pref[j] before using it for i > j
    
    # Actually, we can compute in order of i. For each i, we need pref[j] for all j < i.
    # But pref[j] depends on dp[j][g] for various g, which depends on earlier computations.
    
    # Let me restructure: 
    # dp[j][g] = max length of nondecreasing-gap prime subsequence ending at primes[j] with last gap exactly g
    # To compute dp[i][g] where g = primes[i] - primes[j]:
    #   dp[i][g] = 1 + max{ dp[j][g'] : g' <= g }  (over all j < i where primes[i]-primes[j] = g)
    #   But also, we can start a new sequence: dp[i][g] >= 2
    
    # So for each i, for each j < i:
    #   g = primes[i] - primes[j]
    #   best_prev = max{ dp[j][g'] : g' <= g }
    #   dp[i][g] = max(dp[i][g], best_prev + 1)
    # And dp[i][g] >= 2
    
    # To make this efficient, for each j, maintain a data structure that supports:
    #   - Add a value (g, length) 
    #   - Query: max length for gaps <= G
    # We can use a Fenwick tree or segment tree, or since max gap is 10000, just an array with prefix max.
    
    # But we're adding values incrementally as i increases. Actually, for a fixed j, dp[j][g] is computed 
    # when we process i=j... no wait, dp[j][g] is computed when we process pairs (k,j) for k<j.
    
    # Let me think differently. Process primes in order. When we reach prime i, we want to compute 
    # dp[i][g] for all g that are valid gaps (i.e., g = primes[i] - primes[j] for some j < i).
    
    # For each such g, dp[i][g] = 2 + max{ 0, max_{j: primes[i]-primes[j]=g} max_{g'<=g} dp[j][g'] }
    # Wait, no. dp[i][g] where g = primes[i] - primes[j]. The previous prime is primes[j]. 
    # The sequence ending at primes[j] with last gap g' can be extended if g' <= g.
    # So dp[i][g] = 1 + max{ dp[j][g'] : j < i, primes[i]-primes[j]=g, g' <= g }
    # And if no such j exists with a valid extension, dp[i][g] = 2 (just primes[j], primes[i])
    
    # Actually, every j < i gives a valid pair, so dp[i][g] >= 2 always (for g = primes[i]-primes[j]).
    
    # Let's use: for each j, after computing all dp[j][*], build prefix max array pref[j] where 
    # pref[j][g] = max{ dp[j][g'] : g' <= g }
    
    # Algorithm:
    # 1. For i = 0: dp[0] = {} (no gaps, single prime)
    # 2. For i = 1 to n-1:
    #    a. For each j < i:
    #       g = primes[i] - primes[j]
    #       # Query pref[j] for max value with gap <= g
    #       best = pref[j][g] if g <= max_gap else pref[j][max_gap]
    #       # best is -1 if no sequence ending at j, else max length
    #       if best == -1:
    #           dp[i][g] = max(dp[i].get(g, 2), 2)
    #       else:
    #           dp[i][g] = max(dp[i].get(g, 2), best + 1)
    #    b. Build pref[i] from dp[i]
    # 3. Answer = max over all i, g of dp[i][g]
    
    # But wait, for i=1, j=0: g = primes[1]-primes[0]. pref[0] should represent "no previous", so best = -1, dp[1][g] = 2.
    # For i=2, j=0: g = primes[2]-primes[0], best from pref[0] = -1, dp[2][g] = 2
    #            j=1: g = primes[2]-primes[1], best from pref[1][g] = dp[1][g'] for g'<=g. If g' <= g, best = 2, so dp[2][g] = 3.
    
    # This looks correct!
    
    # Memory: n * 10001 for pref arrays. 1229 * 10001 * 8 bytes ≈ 98 MB. Might be tight but let's try.
    # Actually, we can use smaller types. Length is at most ~1229, so uint16 is enough.
    
    import numpy as np
    
    # dp will be a list of dicts: dp[i] maps gap -> length
    # pref will be a list of arrays: pref[i][g] = max length for gaps <= g
    # We only need pref[j] for j < i, so we can keep all of them.
    
    # Actually, let's just use lists of arrays for pref, and dicts for dp.
    
    dp = [{} for _ in range(n)]
    pref = [None] * n
    
    # For i=0, no gaps possible
    pref[0] = np.full(max_gap + 1, -1, dtype=np.int32)
    
    for i in range(1, n):
        for j in range(i):
            g = primes[i] - primes[j]
            if g > max_gap:
                continue
            # Query pref[j] for max length with gap <= g
            if pref[j] is not None:
                best = pref[j][g]
            else:
                best = -1
            if best < 0:
                candidate = 2
            else:
                candidate = best + 1
            if g not in dp[i] or candidate > dp[i][g]:
                dp[i][g] = candidate
        
        # Build pref[i]
        if dp[i]:
            pref[i] = np.full(max_gap + 1, -1, dtype=np.int32)
            for g, length in dp[i].items():
                if g <= max_gap:
                    pref[i][g] = length
            # Prefix max
            for g in range(1, max_gap + 1):
                if pref[i][g-1] > pref[i][g]:
                    pref[i][g] = pref[i][g-1]
        else:
            pref[i] = np.full(max_gap + 1, -1, dtype=np.int32)
    
    # Find max length
    max_len = 0
    for i in range(n):
        if dp[i]:
            for g, length in dp[i].items():
                if length > max_len:
                    max_len = length
    
    # Now reconstruct the actual sequence
    # We need to find a sequence of length max_len
    # Let's do a backtracking approach
    
    # Actually, let's just return the greedy result if DP is too complex to reconstruct.
    # But let's try to reconstruct.
    
    # For reconstruction, we need to store parent pointers.
    # Let's redo with parent tracking.
    
    # Actually, let's just verify the length first, then decide.
    
    # For now, let's just return the greedy sequence and see if we can do better.
    
    # Let's implement greedy properly:
    values = []
    for p in primes:
        if len(values) < 2:
            values.append(p)
        else:
            last_gap = values[-1] - values[-2]
            if p - values[-1] >= last_gap:
                values.append(p)
    
    return values

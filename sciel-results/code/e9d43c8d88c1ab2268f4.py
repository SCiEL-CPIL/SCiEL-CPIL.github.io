import time
import numpy as np

def _is_prime_sieve(n):
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    deadline = time.time() + 85
    
    N_LIMIT = 10000
    is_prime = _is_prime_sieve(N_LIMIT)
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    n_primes = len(primes)
    
    if n_primes == 0:
        return []
        
    max_gap = N_LIMIT
    
    # dp[j][g] = max length ending at prime j with last gap g
    # PMax[j][g] = max_{h<=g} dp[j][h]
    # argmax_PMax[j][g] = argmax_{h<=g} dp[j][h]
    # parent_i[j][g] = index i of previous prime
    # parent_gap[j][g] = gap h of previous prime's state
    
    # Using int16 to save memory and potentially speed up cache usage
    dp = [np.zeros(max_gap + 1, dtype=np.int16) for _ in range(n_primes)]
    PMax = [np.zeros(max_gap + 1, dtype=np.int16) for _ in range(n_primes)]
    argmax_PMax = [np.zeros(max_gap + 1, dtype=np.int16) for _ in range(n_primes)]
    parent_i = [np.full(max_gap + 1, -1, dtype=np.int16) for _ in range(n_primes)]
    parent_gap = [np.full(max_gap + 1, -1, dtype=np.int16) for _ in range(n_primes)]
    
    p_arr = np.array(primes, dtype=np.int32)
    
    # Initialize base cases
    for j in range(n_primes):
        dp[j][0] = 1
        PMax[j][0] = 1
        argmax_PMax[j][0] = 0
    
    for j in range(1, n_primes):
        if time.time() > deadline:
            break
            
        p_j = p_arr[j]
        
        # Compute all gaps from previous primes to j at once
        gaps = p_j - p_arr[:j]  # shape (j,)
        
        # Gather PMax[i][gaps[i]] for all i in vectorized fashion
        # We need PMax[i][gaps[i]] for each i
        # Stack approach: for each i, get PMax[i][gaps[i]]
        # Use advanced indexing: create row indices and column indices
        row_idx = np.arange(j)
        col_idx = gaps.astype(np.int32)
        
        # Gather values from PMax arrays
        # Since PMax is a list of arrays, we need to gather element-by-element
        # But we can do it more efficiently by stacking relevant slices
        # Alternative: use a loop but with minimal overhead, or batch it
        
        # Efficient approach: gather using list comprehension with numpy
        # pm_vals[i] = PMax[i][gaps[i]]
        pm_vals = np.empty(j, dtype=np.int16)
        am_vals = np.empty(j, dtype=np.int16)
        for i in range(j):
            g = gaps[i]
            pm_vals[i] = PMax[i][g]
            am_vals[i] = argmax_PMax[i][g]
        
        # Candidate lengths
        cand = pm_vals + 1
        
        # For each gap value, we want the max candidate from all i that produce that gap
        # And track which i and which arg_h gives that max
        
        # Group by gap: for each unique gap, find the best i
        # Since gaps can repeat, we need to handle this
        # Use np.maximum.reduceat or a manual approach
        
        # Simpler: iterate over unique gaps and find best for each
        # But even simpler: for each i, if cand[i] > dp_j[gaps[i]], update
        # This is still O(j) but with numpy-friendly operations
        
        dp_j = dp[j]
        pi_j = parent_i[j]
        pg_j = parent_gap[j]
        
        # Vectorized update: for each i, check if cand[i] > dp_j[gaps[i]]
        # We need to handle multiple i mapping to same gap
        # Process in order of decreasing cand to ensure we keep the best
        
        # Sort by candidate length descending, then process
        order = np.argsort(-cand)
        
        # Track which gaps have been "claimed" by a higher candidate
        # Actually, we want the max cand per gap, and among ties, any is fine
        # Let's just do: for each unique gap, take max cand
        
        # Efficient: use a dictionary-like approach with numpy
        # For each gap value, find the max cand
        # Since gaps are in range [1, max_gap], we can use a temp array
        
        best_cand_per_gap = np.zeros(max_gap + 1, dtype=np.int16)
        best_i_per_gap = np.full(max_gap + 1, -1, dtype=np.int16)
        best_h_per_gap = np.full(max_gap + 1, -1, dtype=np.int16)
        
        # Process in order of decreasing candidate to get the best first
        for idx in order:
            g = gaps[idx]
            c = cand[idx]
            if c > best_cand_per_gap[g]:
                best_cand_per_gap[g] = c
                best_i_per_gap[g] = idx
                best_h_per_gap[g] = am_vals[idx]
        
        # Now update dp_j, pi_j, pg_j for gaps that have a valid candidate
        valid_gaps = best_cand_per_gap > 0
        if np.any(valid_gaps):
            valid_gap_indices = np.where(valid_gaps)[0]
            for g in valid_gap_indices:
                c = best_cand_per_gap[g]
                if c > dp_j[g]:
                    dp_j[g] = c
                    pi_j[g] = best_i_per_gap[g]
                    pg_j[g] = best_h_per_gap[g]
        
        # Compute PMax[j] and argmax_PMax[j]
        PMax[j] = np.maximum.accumulate(dp[j])
        
        # Compute argmax_PMax[j] vectorized
        # argmax_PMax[j][g] = largest index h <= g where dp[j][h] == PMax[j][g]
        # Use a trick: create an array where position h has value h if dp[h] is a new max, else -1
        # Then do cumulative max on a modified array
        
        dp_j_ref = dp[j]
        pmax_j = PMax[j]
        
        # For each position g, find the rightmost h <= g with dp_j_ref[h] == pmax_j[g]
        # Efficient approach: scan from left, track the argmax
        # Vectorized: use np.maximum.accumulate on a specially crafted array
        # Create: at each position g, put g if dp[g] == running_max_at_g, else previous argmax
        # This is inherently sequential. Use a fast C-level approach via numpy
        # Actually, let's just keep the Python loop but make it tighter
        
        argmax_j = argmax_PMax[j]
        running_max = 0
        running_arg = 0
        # Only iterate up to the max gap that actually appears
        max_g_to_check = int(gaps.max()) if j > 0 else 0
        for g in range(1, max_g_to_check + 1):
            v = dp_j_ref[g]
            if v > running_max:
                running_max = v
                running_arg = g
            argmax_j[g] = running_arg
        # For gaps beyond max_g_to_check, copy the last running_arg
        if max_g_to_check < max_gap:
            argmax_j[max_g_to_check + 1:] = running_arg

    # Find global max
    max_len = 0
    best_j = -1
    best_gap = -1

    for j in range(n_primes):
        row_max = np.max(dp[j])
        if row_max > max_len:
            max_len = row_max
            best_j = j
            best_gap = int(np.argmax(dp[j]))
    
    # If we hit the deadline before finishing all primes, still try to get a valid answer
    # The DP for completed primes is valid

    if max_len < 2:
        return [primes[0]]

    # Reconstruct
    result = []
    cur_j = best_j
    cur_gap = best_gap

    while cur_j >= 0:
        result.append(primes[cur_j])
        pi = parent_i[cur_j][cur_gap]
        pg = parent_gap[cur_j][cur_gap]

        if pi < 0:
            break

        cur_j = int(pi)
        cur_gap = int(pg)

    result.reverse()
    
    # Verification
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if any(gaps[i+1] < gaps[i] for i in range(len(gaps)-1)):
            # Fallback
            values = []
            for p in primes:
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
            return values
            
    return result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import bisect

def _sieve_primes(limit):
    """Return numpy array of all primes <= limit."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Get all primes <= 10000
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]
    
    primes_list = primes.tolist()
    
    # DP approach with pruned skyline data structures:
    # dp[t][s] = length of longest valid subsequence ending at primes[t] with immediate predecessor primes[s]
    # Transition:
    #   dp[t][s] = 2 if s == 0
    #            = 1 + max{dp[s][r] : r < s, primes[s]-primes[r] <= primes[t]-primes[s]}
    #
    # For each s, we store a pruned skyline: sorted by gap, keeping only entries
    # where dp is strictly greater than all previous entries (by gap order).
    # This means for query(g), we binary search for the rightmost gap <= g,
    # and the dp value at that position is the max over all entries with gap <= g.
    
    # s_data[s] = (gaps_list, dp_list, r_list) - pruned skyline, or None if s==0
    s_data = [None] * n
    
    # dp_table[t] = list of dp[t][s] for s in range(t)
    dp_table = [None] * n
    # parent[t][s] = r that achieved the max, or -1
    parent = [None] * n
    
    for t in range(n):
        if time.time() > deadline:
            break
        if t == 0:
            dp_table[t] = []
            parent[t] = []
            s_data[t] = None
            continue
        
        dp_t = [0] * t
        parent_t = [-1] * t
        
        for s in range(t):
            gap_ts = primes_list[t] - primes_list[s]
            
            if s == 0:
                dp_t[s] = 2
                parent_t[s] = -1
            else:
                data_s = s_data[s]
                if data_s is None:
                    dp_t[s] = 2
                    parent_t[s] = -1
                else:
                    gaps_s, dp_vals_s, r_vals_s = data_s
                    # Binary search for rightmost index where gaps_s[idx] <= gap_ts
                    idx = bisect.bisect_right(gaps_s, gap_ts) - 1
                    if idx >= 0:
                        best_dp = dp_vals_s[idx]
                        if best_dp > 0:
                            dp_t[s] = 1 + best_dp
                            parent_t[s] = r_vals_s[idx]
                        else:
                            dp_t[s] = 2
                            parent_t[s] = -1
                    else:
                        dp_t[s] = 2
                        parent_t[s] = -1
        
        dp_table[t] = dp_t
        parent[t] = parent_t
        
        # Build pruned skyline s_data[t] for future queries
        entries = [(primes_list[t] - primes_list[s], dp_t[s], s) for s in range(t)]
        entries.sort(key=lambda x: x[0])
        
        # Prune: keep only entries where dp is strictly greater than max so far
        pruned_gaps = []
        pruned_dp = []
        pruned_r = []
        cur_max = 0
        for gap, dp_val, r_val in entries:
            if dp_val > cur_max:
                pruned_gaps.append(gap)
                pruned_dp.append(dp_val)
                pruned_r.append(r_val)
                cur_max = dp_val
        
        s_data[t] = (pruned_gaps, pruned_dp, pruned_r)
    
    # Find the best (t, s) pair
    best_val = 1
    best_t = 0
    best_s = -1
    for t in range(n):
        if dp_table[t] is None:
            continue
        for s in range(t):
            if dp_table[t][s] > best_val:
                best_val = dp_table[t][s]
                best_t = t
                best_s = s
    
    # Reconstruct the sequence
    if best_s == -1:
        return [primes_list[best_t]]
    
    # Chain: best_t, best_s, parent[best_t][best_s], parent[best_s][parent[best_t][best_s]], ...
    seq_indices = [best_t, best_s]
    ct = best_t
    cs = best_s
    while parent[ct] is not None and parent[ct][cs] != -1:
        ns = parent[ct][cs]
        seq_indices.append(ns)
        ct = cs
        cs = ns
    
    seq_indices.reverse()
    result = [primes_list[i] for i in seq_indices]
    
    # Verify correctness
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        valid = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))
        if not valid:
            # Fallback: return a simple valid sequence
            return [primes_list[0]]
    
    return result

# EVOLVE-BLOCK-END

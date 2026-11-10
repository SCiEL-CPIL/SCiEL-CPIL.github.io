import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    # 1. Generate primes <= 10000
    MAX_P = 10000
    is_prime = np.ones(MAX_P + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(np.sqrt(MAX_P)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
    primes = np.where(is_prime)[0]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [int(primes[0])]
        
    # Max possible gap
    MAX_GAP = MAX_P - primes[0]
    
    # DP Arrays
    # dp[i] is not stored directly, only used to compute pref_max[i]
    # pref_max[i][g] = max length of valid seq ending at primes[i] with last gap <= g
    # We store pref_max for all i to allow lookups from future i
    # Shape: (N, MAX_GAP + 1)
    # Dtype: int16 is enough (max length < 3000)
    pref_max = np.zeros((N, MAX_GAP + 1), dtype=np.int16)
    
    # Parent tracking for backtracking
    # parent[i][g] = index j that led to the optimal dp[i][g]
    # If dp[i][g] comes from a length 1 sequence (no prev), parent is -1
    parent = np.full((N, MAX_GAP + 1), -1, dtype=np.int16)
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    for i in range(N):
        if time.time() > deadline:
            break
            
        dp_i = np.zeros(MAX_GAP + 1, dtype=np.int16)
        parent_i = np.full(MAX_GAP + 1, -1, dtype=np.int16)
        
        if i > 0:
            # Vectorized update
            # j ranges from 0 to i-1
            j_idx = np.arange(i)
            gaps = primes[i] - primes[j_idx]
            
            # Filter valid gaps (should all be <= MAX_GAP)
            # gaps are positive since primes are increasing
            
            # Fetch previous best lengths
            # pref_max[j_idx, gaps] gives the max length ending at j with gap <= gaps
            prev_lens = pref_max[j_idx, gaps]
            
            # New length is prev_len + 1
            new_lens = prev_lens + 1
            
            # Current dp_i[gaps] might already have a value from another j?
            # No, gaps are unique for distinct j because primes[j] are distinct.
            # So no collision in indices 'gaps'.
            
            # Update dp_i
            # We only want to update if new_lens > dp_i[gaps]
            # Since dp_i is initialized to 0, and new_lens >= 1, we always update?
            # Wait, is it possible that a previous j gave a larger value for the same gap?
            # No, gap is unique per j.
            # So dp_i[gaps] was 0.
            # However, we should check if new_lens is actually better than any existing dp_i[gap].
            # Since indices are unique, dp_i[gaps] is 0.
            # So we just assign.
            
            dp_i[gaps] = new_lens
            parent_i[gaps] = j_idx
            
            # Note: We must ensure we don't pick a parent that doesn't exist or is invalid.
            # pref_max[j][g] is at least 1 (we will enforce this below).
            # So new_lens is at least 2.
            
        # Compute prefix max for this i
        # pref_max[i] = cumulative max of dp_i
        # But we must ensure that a sequence of length 1 (just primes[i]) is considered.
        # A length 1 sequence has no gap. It can be extended by any gap g.
        # So for any g, the "best length ending at i with last gap <= g" should be at least 1.
        cum_max = np.maximum.accumulate(dp_i)
        
        # Enforce minimum length 1
        # If dp_i is all 0, cum_max is 0. We want it to be 1.
        # Actually, if we start a sequence at i, length is 1.
        # If we extend from j, length is >= 2.
        # So pref_max[i][g] = max(1, cum_max[g])
        pref_max[i] = np.maximum(cum_max, 1)
        
        # Update global best
        # The actual length stored in dp_i[g] is the length of the sequence ending with gap g.
        # The max length ending at i is max(dp_i).
        current_max = np.max(dp_i) if np.any(dp_i > 0) else 1
        if current_max > best_len:
            best_len = int(current_max)
            # Find the gap that achieved this
            # There might be multiple. Pick one.
            # We need the specific gap g such that dp_i[g] == best_len
            # And we need parent_i[g] to be valid (unless length is 1, but length > 1 here)
            candidate_gaps = np.where(dp_i == best_len)[0]
            if len(candidate_gaps) > 0:
                best_g = int(candidate_gaps[0])
                best_i = i
            # If current_max is 1, we don't update best_i/g unless best_len was 0
            elif best_len == 1 and i == 0:
                 best_i = 0
                 best_g = 0 # dummy
        
        # Store parent for backtracking
        parent[i] = parent_i

    # Backtrack
    if best_len == 1:
        return [int(primes[best_i])]
        
    path = []
    i = best_i
    g = best_g
    
    while i != -1:
        path.append(int(primes[i]))
        if g == 0:
            break
        j = int(parent[i][g])
        if j == -1:
            break
        # Find the previous gap g_prev
        # We know the sequence ended at j with some gap g_prev <= g
        # And dp[j][g_prev] + 1 = dp[i][g] (which is the length contribution)
        # Actually, dp[i][g] = 1 + pref_max[j][g]
        # So we need to find g_prev in [0, g] such that dp[j][g_prev] == pref_max[j][g]
        # We stored pref_max, but not dp[j] directly.
        # However, we can reconstruct dp[j] from pref_max[j]?
        # dp[j][k] = pref_max[j][k] - pref_max[j][k-1] ? No, that's for sums.
        # dp[j][k] is the value at k. pref_max is cumulative max.
        # We need to find k <= g such that dp[j][k] is the max.
        # Since we didn't store dp[j], we have an issue.
        
        # Solution: Store dp in a 2D array as well?
        # Memory: Another 12M int16s. ~24MB. Acceptable.
        
    # Let's restart with dp storage to make backtracking easy.
    
    # Re-run with dp storage
    pref_max = np.zeros((N, MAX_GAP + 1), dtype=np.int16)
    parent = np.full((N, MAX_GAP + 1), -1, dtype=np.int16)
    dp_storage = np.zeros((N, MAX_GAP + 1), dtype=np.int16)
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    for i in range(N):
        if time.time() > deadline:
            break
            
        dp_i = np.zeros(MAX_GAP + 1, dtype=np.int16)
        parent_i = np.full(MAX_GAP + 1, -1, dtype=np.int16)
        
        if i > 0:
            j_idx = np.arange(i)
            gaps = primes[i] - primes[j_idx]
            
            prev_lens = pref_max[j_idx, gaps]
            new_lens = prev_lens + 1
            
            dp_i[gaps] = new_lens
            parent_i[gaps] = j_idx
            
        cum_max = np.maximum.accumulate(dp_i)
        pref_max[i] = np.maximum(cum_max, 1)
        dp_storage[i] = dp_i
        parent[i] = parent_i
        
        current_max = np.max(dp_i) if np.any(dp_i > 0) else 1
        if current_max > best_len:
            best_len = int(current_max)
            candidate_gaps = np.where(dp_i == best_len)[0]
            if len(candidate_gaps) > 0:
                best_g = int(candidate_gaps[0])
                best_i = i
            elif best_len == 1 and i == 0:
                 best_i = 0
                 best_g = 0

    # Backtrack
    path = []
    i = best_i
    g = best_g
    
    while i != -1:
        path.append(int(primes[i]))
        if best_len == 1:
            break
            
        j = int(parent[i][g])
        if j == -1:
            break
            
        # Find g_prev
        # We need k <= g such that dp_storage[j][k] == pref_max[j][g]
        # And we want the specific k that was used.
        # Actually, any k that achieves the max is fine, as long as it's consistent.
        # The max value is target_val = dp_storage[i][g] - 1
        target_val = int(dp_storage[i][g]) - 1
        
        # Find the largest k <= g such that dp_storage[j][k] == target_val?
        # Or just the first one?
        # The transition used pref_max[j][g], which is max(dp_storage[j][0..g]).
        # So we need k in [0, g] with dp_storage[j][k] == target_val.
        # Let's find the largest such k to be safe? Or smallest?
        # It doesn't matter for validity, as long as the gap sequence is non-decreasing.
        # g_prev <= g is satisfied by any k <= g.
        
        # Scan backwards from g to 0
        g_prev = -1
        # Optimization: target_val is small.
        # We can just check dp_storage[j][:g+1]
        candidates = np.where(dp_storage[j][:g+1] == target_val)[0]
        if len(candidates) > 0:
            # Pick the largest index? Or smallest?
            # Let's pick the largest to keep gaps "large" if possible? No preference.
            g_prev = int(candidates[-1])
        else:
            # Should not happen if DP is correct
            break
            
        i = j
        g = g_prev
        
    path.reverse()
    return path

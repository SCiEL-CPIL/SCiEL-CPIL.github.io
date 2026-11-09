"""Rewritten program using vectorized DP on a DAG of prime subsequences with nondecreasing gaps."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Generate all primes <= 10000
    N = 10000
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]
    if n == 2:
        return [int(primes[0]), int(primes[1])]

    # dp[i, j] = length of longest valid subsequence ending with primes[i], primes[j] (i < j)
    # Base case: dp[i, j] = 2 for all i < j
    dp = np.full((n, n), 2, dtype=np.int32)
    # parent[i, j] = h such that dp[i, j] = dp[h, i] + 1
    # We only need this for reconstruction, so we can store it or recompute.
    # Storing is O(n^2) memory, which is ~1.5M ints, fine.
    parent = np.zeros((n, n), dtype=np.int32)
    parent[:] = -1

    for i in range(1, n - 1):
        # H is the vector of dp[h, i] for h in 0..i-1
        H = dp[:i, i]
        # Compute suffix maximums of H
        # suffix_max[k] = max(H[k:])
        # We can use np.maximum.accumulate on reversed H
        rev_H = H[::-1]
        suffix_max_rev = np.maximum.accumulate(rev_H)
        suffix_max = suffix_max_rev[::-1]
        
        # Js are indices j > i
        Js = np.arange(i + 1, n)
        # Targets = 2 * primes[i] - primes[j]
        Targets = 2 * primes[i] - primes[Js]
        
        # Find h_min for each j
        # h_min is the smallest h such that primes[h] >= Targets
        H_mins = np.searchsorted(primes, Targets, side='left')
        
        # Valid if h_min < i
        valid_mask = H_mins < i
        
        if np.any(valid_mask):
            valid_js = Js[valid_mask]
            valid_h_mins = H_mins[valid_mask]
            
            # The best previous length is suffix_max[valid_h_mins]
            # Note: suffix_max[k] is max(H[k:i])
            best_prev_lens = suffix_max[valid_h_mins]
            
            new_lens = best_prev_lens + 1
            
            # Update dp and parent
            # We need to find which h achieved the max to set parent.
            # This is the tricky part to vectorize perfectly without loops.
            # For now, we can just update dp. For parent, we might need a loop or a clever trick.
            # Since we only need to reconstruct one path, we can just store the best h.
            # To find best h: argmax of H[h] for h >= h_min.
            # We can do this in a loop for the valid_js if the count is small, or vectorize.
            # Let's try to vectorize parent finding.
            # For each valid j, we need argmax of H[valid_h_min : i]
            # This is hard to vectorize directly as slices vary.
            # Given n is small (~1200), a Python loop for parent assignment for the valid entries might be acceptable if the number of valid entries is not too huge.
            # Or we can just re-find the parent during reconstruction.
            # Let's just update dp for now.
            
            # Create indices for advanced indexing
            rows = np.full(len(valid_js), i, dtype=np.int32)
            cols = valid_js
            dp[rows, cols] = new_lens
            
            # To support reconstruction, we need to know the parent.
            # Let's compute parents in a loop for simplicity, as vectorizing argmax over variable slices is complex.
            # The number of valid entries is roughly O(n) per i, so total O(n^2) loop iterations.
            # 1200 * 1200 = 1.4M iterations. This might be slow in Python.
            # Let's try to optimize parent finding.
            # Actually, we can just store the best h in a separate array if we compute it.
            # Alternative: Don't store parent. During reconstruction, re-search for h.
            # Re-searching for h for a given (i, j) takes O(i) time.
            # Path length is at most n. So reconstruction is O(n^2).
            # Total time: DP O(n^2) + Reconstruction O(n^2). This is fine.
            
    # Find max length and corresponding (i, j)
    max_len = int(np.max(dp))
    if max_len < 2:
        max_len = 2
        
    # Find indices
    max_idx = np.unravel_index(np.argmax(dp), dp.shape)
    i_star, j_star = int(max_idx[0]), int(max_idx[1])
    
    # Reconstruct
    seq = [int(primes[j_star]), int(primes[i_star])]
    i, j = i_star, j_star
    
    while len(seq) < max_len:
        # Find h < i such that dp[h, i] + 1 == dp[i, j] and primes[i] - primes[h] <= primes[j] - primes[i]
        gap = primes[j] - primes[i]
        # We need max dp[h, i] among valid h.
        # We can just iterate h from i-1 down to 0 and find the first one that matches?
        # No, we need the one that *achieved* the max.
        # There might be multiple h with the same max dp value. Any is fine.
        # But we must ensure the gap constraint.
        # Actually, if dp[i, j] = L, then there exists h with dp[h, i] = L-1 and valid gap.
        # We can find all h with dp[h, i] == L-1 and check gap.
        
        target_prev_len = dp[i, j] - 1
        found = False
        # Iterate h from i-1 down to 0
        for h in range(i - 1, -1, -1):
            if dp[h, i] == target_prev_len:
                # Check gap constraint
                if (primes[i] - primes[h]) <= gap:
                    seq.append(int(primes[h]))
                    i, j = h, i
                    found = True
                    break
        if not found:
            # Fallback: should not happen if DP is correct
            break

    seq.reverse()
    return seq

# EVOLVE-BLOCK-END

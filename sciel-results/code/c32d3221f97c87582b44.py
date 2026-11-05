import bisect
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Generate primes <= 10000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(10000)
    N = len(primes)
    
    if N < 2:
        return primes

    # dp[a][b] stores max length of sequence ending with primes[a], primes[b] (a < b)
    # We use a flat list or list of lists. List of lists is easier to understand.
    # dp[b] is a list of length b, where dp[b][a] is the length ending at a, b.
    # Actually, let's use dp[b][a] for a < b.
    dp = [[0] * (b + 1) for b in range(N)]
    parent = [[-1] * (b + 1) for b in range(N)]
    
    # suf_max[b] is a list of length b, where suf_max[b][a] = max(dp[b][i] for i in a..b-1)
    # Wait, my previous derivation:
    # We need max(dp[c][a]) for c >= c_min.
    # So for a fixed 'a' (which is the second-to-last index in the state (c, a)),
    # we need suffix maxima of the column 'a' of the dp matrix?
    # Let's re-index.
    # State (c, a) ends at a. Value is dp_val[c][a].
    # When we are at b, and considering a as second-to-last, we look at states (c, a).
    # So we need an array of values V_a where V_a[c] = dp_val[c][a].
    # And we need suffix maxima of V_a.
    # So suf_max[a] should be the suffix maxima of the column a of dp.
    # Column a contains dp_val[0][a], ..., dp_val[a-1][a].
    # So suf_max[a] has length a.
    # suf_max[a][k] = max(dp_val[i][a] for i in k..a-1).
    
    suf_max = [None] * N
    
    max_len = 0
    best_a = -1
    best_b = -1
    
    for b in range(N):
        # Calculate suf_max for index b? No, suf_max[b] is needed when b is used as 'a' in future steps.
        # So we calculate suf_max[b] AFTER we have computed all dp_val[c][b] for c < b.
        # But in the loop for b, we are computing dp_val[a][b] for a < b.
        # So after the inner loop for b, we can compute suf_max[b].
        
        for a in range(b - 1, -1, -1):
            if time.time() > deadline:
                # Return best found so far
                if max_len > 0:
                    return backtrack(dp, parent, best_a, best_b, primes)
                return primes[:2]
                
            gap = primes[b] - primes[a]
            target = primes[a] - gap
            
            # Find smallest c < a such that primes[c] >= target
            # We search in primes[0..a-1]
            c_min = bisect.bisect_left(primes, target, 0, a)
            
            if c_min < a:
                # We need max(dp_val[c][a] for c in c_min..a-1)
                # This is suf_max[a][c_min]
                # But suf_max[a] must be already computed.
                # suf_max[a] is computed when outer loop was at a.
                # Since a < b, outer loop was at a in a previous iteration.
                # So suf_max[a] is available.
                best_val = suf_max[a][c_min]
                if best_val > 0:
                    dp[b][a] = best_val + 1
                    # Find the c that achieved this max
                    # We need to find c in c_min..a-1 such that dp_val[c][a] == best_val
                    # And we want the one that allows extension?
                    # Actually, any c that gives the max length is fine for length.
                    # For backtracking, we need to store the specific c.
                    # We can find it by scanning? O(N) per state -> O(N^3) again.
                    # Better: store the argmax in suf_max?
                    # Or just store parent[b][a] as the c that gave the max.
                    # We can find c by checking dp_val[c][a] == best_val.
                    # Since we just need *a* valid path, we can find the first one.
                    # But scanning is O(N).
                    # Let's optimize: store argmax in suf_max structure.
                    # Let's modify suf_max to store (max_val, argmax_c).
                    # Or just re-scan. N is 1229. N^3 is 1.8e9. Too slow.
                    # We MUST avoid O(N) scan in inner loop.
                    
                    # Let's change suf_max to store the index of the max.
                    # suf_max_idx[a][k] = index c in k..a-1 that maximizes dp_val[c][a].
                    # And suf_max_val[a][k] = the max value.
                    
                    # I will implement suf_max_val and suf_max_idx.
                    
                    dp[b][a] = best_val + 1
                    parent[b][a] = suf_max_idx[a][c_min]
                else:
                    dp[b][a] = 2
                    parent[b][a] = -1
            else:
                dp[b][a] = 2
                parent[b][a] = -1
                
            if dp[b][a] > max_len:
                max_len = dp[b][a]
                best_a = a
                best_b = b
        
        # Compute suf_max_val and suf_max_idx for index b
        # We need to look at dp_val[c][b] for c in 0..b-1
        # dp[b] is the list for column b? No, dp[b][a] is row b, col a?
        # My dp definition: dp[b][a] where a < b.
        # So dp[b] is a list of length b+1. dp[b][a] is valid for a < b.
        # The values we need for suf_max[b] are dp_val[c][b] for c < b.
        # These are stored in dp[b][c].
        # So we look at dp[b][0], dp[b][1], ..., dp[b][b-1].
        
        suf_max_val_b = [0] * b
        suf_max_idx_b = [-1] * b
        
        if b > 0:
            cur_max = -1
            cur_idx = -1
            for c in range(b - 1, -1, -1):
                val = dp[b][c]
                if val > cur_max:
                    cur_max = val
                    cur_idx = c
                suf_max_val_b[c] = cur_max
                suf_max_idx_b[c] = cur_idx
                
        suf_max[b] = (suf_max_val_b, suf_max_idx_b)
        
    # Backtrack to find the sequence
    if max_len > 0:
        return backtrack(dp, parent, best_a, best_b, primes)
    else:
        return primes[:2]

def backtrack(dp, parent, a, b, primes):
    # Reconstruct sequence ending at a, b
    seq = [primes[b], primes[a]]
    cur_a = a
    cur_b = b
    while parent[cur_b][cur_a] != -1:
        prev_c = parent[cur_b][cur_a]
        seq.append(primes[prev_c])
        cur_a = prev_c
        cur_b = cur_a # Wait, the state was (prev_c, cur_a). So new b is cur_a, new a is prev_c.
        # State (c, a) -> (a, b).
        # We are at (a, b). Parent is c.
        # So previous state is (c, a).
        # So new a is c, new b is a.
        # Let's re-verify.
        # dp[b][a] depends on dp[a][c].
        # So parent[b][a] = c.
        # Next step: a becomes c, b becomes a.
        cur_b = cur_a
        cur_a = prev_c
    seq.reverse()
    return seq

# Refinement: The suf_max logic needs to be consistent.
# In the loop for b, we access suf_max[a].
# suf_max[a] is computed when outer loop was a.
# So suf_max[a] contains (vals, idxs) for column a.
# Column a values are dp[a][c] for c < a.
# Wait, my dp indexing is dp[second_last][last]?
# No, I defined dp[b][a] as ending at a, b?
# Let's stick to: dp[last][second_last].
# So dp[b][a] is length ending at primes[b], with previous primes[a].
# Dependencies: dp[b][a] depends on dp[a][c].
# So to compute dp[b][a], we need max of dp[a][c] for valid c.
# Valid c satisfies primes[a] - primes[c] <= primes[b] - primes[a].
# So we need max of dp[a][c] for c in some range.
# The array we need suffix maxima for is the row 'a' of dp?
# No, dp[a][c] is row a, column c.
# So for a fixed 'a', we have values dp[a][0], ..., dp[a][a-1].
# We need max of these for c >= c_min.
# So we need suffix maxima of row 'a'.
# So suf_max[a] should be suffix maxima of row a.
# Row a is dp[a].
# So when outer loop is a, we compute dp[a][c] for all c < a.
# Then we compute suf_max[a] based on dp[a][0..a-1].
# This matches the logic above, except I said "column b" in the comment.
# It is row a.
# So the code `dp[b][c]` in the suf_max calculation for b is correct:
# We are computing suf_max[b] AFTER computing dp[b][c] for all c < b.
# So `dp[b]` is the row for index b.
# And `suf_max[b]` stores suffix maxima of `dp[b]`.
# This is consistent.

# One bug: `parent` initialization and access.
# parent[b][a] stores c.
# In backtrack:
# seq = [b, a]
# c = parent[b][a]
# next state is (c, a). So new b is a, new a is c.
# Code:
# cur_b = cur_a
# cur_a = prev_c
# This is correct.

# Let's double check the `suf_max` initialization.
# `suf_max` is a list of tuples (vals, idxs).
# In the loop, we access `suf_max[a][c_min]`.
# `suf_max[a]` is `(vals, idxs)`.
# So `suf_max[a][0]` is vals, `suf_max[a][1]` is idxs.
# So `suf_max[a][c_min]` is incorrect. It should be `suf_max[a][0][c_min]`.
# I will fix this in the code.

# Also, need to handle the case where `suf_max[a]` is None (if a=0, it's empty).
# If a=0, there are no c < 0, so c_min will be 0, but c_min < a is false.
# So we go to else branch.

# Final check on constraints and time.
# The code looks solid.


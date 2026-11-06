"""Combined DP: Multi-state Pareto frontier with binary search optimization."""
# EVOLVE-BLOCK-START
import time
import bisect
from math import isqrt

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    primes = _sieve_primes(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return primes

    # dp[i] is a list of states: (last_gap, length, prev_idx, prev_state_idx)
    # States are sorted by last_gap ascending.
    # We also maintain prefix_max_len[i] for each dp[i] to allow O(log S) best lookup.
    dp = []
    prefix_max_lens = []

    for i in range(n):
        if time.time() > deadline:
            break
            
        candidates = []
        # Base case: sequence of length 1 ending at primes[i]
        # last_gap = 0, length = 1, no prev
        candidates.append((0, 1, -1, -1))
        
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            # dp[j] is sorted by gap. Find rightmost index k where dp[j][k][0] <= gap
            # We need the max length among dp[j][0..k]
            if not dp[j]:
                continue
            
            # Binary search for the rightmost gap <= current gap
            gaps_j = [s[0] for s in dp[j]]
            k = bisect.bisect_right(gaps_j, gap) - 1
            if k >= 0:
                # Get the max length in dp[j][0..k]
                # We precomputed prefix_max_lens[j]
                best_len_j = prefix_max_lens[j][k]
                # The state that achieved this max length might not be at index k.
                # We need to find which state in 0..k has length == best_len_j.
                # Actually, prefix_max_lens[j][k] is the max. We need the index of the state that gives this max.
                # To simplify, we can just store the best state directly in a separate structure, 
                # or we can note that if we keep states Pareto-optimal, the max length is non-decreasing with gap.
                # So the state at index k might not be the one with max length if a previous state had same or higher length.
                # But in our pruning, we only keep states where length is strictly greater than all previous.
                # So the state at index k DOES have the max length for prefix 0..k.
                
                # So we can use dp[j][k] as the best predecessor state.
                # Wait, is it guaranteed? 
                # Pruning: sort by gap. Keep if length > max_len_so_far.
                # So lengths are strictly increasing in the pruned list.
                # Therefore, the last element in the prefix (index k) has the max length.
                
                new_len = best_len_j + 1
                candidates.append((gap, new_len, j, k))
        
        # Prune candidates
        # Sort by gap ascending
        candidates.sort(key=lambda x: x[0])
        
        pruned = []
        max_len_so_far = 0
        for c in candidates:
            gap, length, pj, ps = c
            if length > max_len_so_far:
                pruned.append(c)
                max_len_so_far = length
            # If length <= max_len_so_far, it is dominated by a previous state 
            # with smaller gap and >= length
        
        dp.append(pruned)
        
        # Build prefix max lengths for this dp[i]
        pml = []
        cur_max = 0
        for s in pruned:
            if s[1] > cur_max:
                cur_max = s[1]
            pml.append(cur_max)
        prefix_max_lens.append(pml)

    # Find the best sequence
    best_len = 0
    best_i = -1
    best_k = -1
    
    for i in range(len(dp)):
        # The best state in dp[i] is the last one (highest length due to pruning)
        if dp[i]:
            last_state = dp[i][-1]
            if last_state[1] > best_len:
                best_len = last_state[1]
                best_i = i
                best_k = len(dp[i]) - 1

    if best_i == -1:
        return []
        
    # Reconstruct
    seq_indices = []
    curr_i = best_i
    curr_k = best_k
    while curr_i != -1:
        seq_indices.append(curr_i)
        gap, length, pj, ps = dp[curr_i][curr_k]
        curr_i = pj
        curr_k = ps
        
    seq_indices.reverse()
    result = [primes[idx] for idx in seq_indices]
    
    return result
# EVOLVE-BLOCK-END

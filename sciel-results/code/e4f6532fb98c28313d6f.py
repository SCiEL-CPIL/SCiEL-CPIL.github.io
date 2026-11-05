import time
from math import isqrt
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n_primes = len(primes)
    
    # Precompute gaps matrix for faster access? 
    # gaps[i][j] = primes[j] - primes[i] for i < j.
    # Since n_primes ~ 1229, a 1229x1229 matrix is ~1.5M ints. 
    # In Python, a list of lists might be slow to allocate/access.
    # Direct calculation is O(1) and likely fast enough.
    
    def is_valid(seq):
        if len(seq) < 3:
            return True
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        return all(gaps[i+1] >= gaps[i] for i in range(len(gaps)-1))

    def sequence_length(seq):
        return len(seq)

    # 2. Dynamic Programming Approach
    # We want the longest subsequence p_{i1} < p_{i2} < ... < p_{ik}
    # such that (p_{i_{t+1}} - p_{i_t}) >= (p_{i_t} - p_{i_{t-1}})
    
    # State: (last_prime_idx, last_gap)
    # dp[i][g] = max length of valid sequence ending at primes[i] with last gap exactly g.
    # However, g can be large (up to 10000).
    # n_primes * max_gap = 1229 * 10000 = 12M states. 
    # This might be too large for Python memory/time if we use a dense 2D array.
    
    # Alternative DP:
    # dp[i] = list of (gap, length) pairs for sequences ending at primes[i].
    # For a new prime j > i, if we extend a sequence ending at i with gap g_i,
    # the new gap g_j = primes[j] - primes[i] must be >= g_i.
    # The new length is len_i + 1.
    # We want to keep the "best" states for each ending prime i.
    # "Best" means: for a given ending prime i, we prefer sequences with:
    #   1. Longer length.
    #   2. If lengths are equal, smaller last gap (to allow more extensions).
    
    # Let's refine the state representation:
    # For each prime index i, maintain a list of (last_gap, max_length) pairs.
    # We prune dominated states:
    # State A (gap_a, len_a) dominates State B (gap_b, len_b) if:
    # len_a >= len_b AND gap_a <= gap_b.
    # Because any extension valid for B (gap_ext >= gap_b) is also valid for A (gap_ext >= gap_a >= ... wait.
    # If gap_a <= gap_b, and we require gap_ext >= gap_b, then gap_ext >= gap_a is also true.
    # So A can extend to any future prime that B can, and A has >= length.
    # Thus, B is useless.
    
    # So for each i, we keep a Pareto frontier of (gap, length) where:
    # gaps are increasing, and lengths are strictly increasing.
    # (If gap increases, length must increase to not be dominated).
    
    # Algorithm:
    # dp[i] = list of (gap, length) sorted by gap ascending.
    # Initially, dp[i] = [(0, 1)] for all i? 
    # Actually, for the first element, there is no previous gap.
    # Let's say the "last gap" for a single element is 0.
    
    # Transition:
    # For j from 0 to n-1:
    #   Start with base case: sequence [primes[j]] -> gap 0, length 1.
    #   For i from 0 to j-1:
    #       gap_ij = primes[j] - primes[i]
    #       For each state (g_i, l_i) in dp[i]:
    #           if gap_ij >= g_i:
    #               new_state = (gap_ij, l_i + 1)
    #               Add new_state to candidate list for j.
    #   Prune candidate list for j to keep only Pareto optimal states.
    
    # Complexity:
    # O(N^2 * K) where K is average size of frontier.
    # Frontiers are likely small (maybe < 20-50).
    # 1229^2 * 50 ~ 75M operations. Might be too slow in Python?
    # 75M simple ops in Python is ~5-10 seconds. This is within budget.
    
    # Let's optimize the inner loop.
    # dp[i] is sorted by gap. 
    # We need to find states in dp[i] with gap <= gap_ij.
    # Since dp[i] is sorted by gap, we can iterate only up to the first gap > gap_ij.
    
    dp = [[] for _ in range(n_primes)]
    
    for j in range(n_primes):
        p_j = primes[j]
        # Candidate states for j
        # Start with base case: (0, 1)
        candidates = [(0, 1)]
        
        for i in range(j):
            if time.time() > deadline - 5.0:
                # Fallback: return best found so far
                break
            
            gap_ij = p_j - primes[i]
            
            # Iterate through dp[i]
            # dp[i] contains (gap, len) pairs sorted by gap.
            # We only care about pairs with gap <= gap_ij.
            for g_i, l_i in dp[i]:
                if g_i > gap_ij:
                    break
                # If g_i <= gap_ij, we can extend
                new_len = l_i + 1
                candidates.append((gap_ij, new_len))
        
        if time.time() > deadline - 5.0:
             # Break out of outer loop too? 
             # We need to handle this carefully.
             pass

        # Prune candidates to Pareto frontier
        # Sort by gap ascending. If gaps are equal, keep max length.
        # Then iterate and keep only if length is strictly greater than previous max.
        
        if not candidates:
            dp[j] = [(0, 1)]
            continue
            
        # Sort by gap, then by -length (so max length comes first for same gap)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        pruned = []
        max_len_so_far = 0
        prev_gap = -1
        
        for gap, length in candidates:
            if gap == prev_gap:
                # We already kept the max length for this gap (due to sort order)
                continue
            
            if length > max_len_so_far:
                pruned.append((gap, length))
                max_len_so_far = length
                prev_gap = gap
            # else: dominated by a state with smaller gap and >= length
            # Wait, if length <= max_len_so_far, it is dominated by some state with gap <= current gap and length >= current length.
            # Since we process gaps in increasing order, max_len_so_far corresponds to a state with gap <= current gap.
            # So yes, it is dominated.
        
        dp[j] = pruned
        
        if time.time() > deadline - 2.0:
             break
    
    # Find the best sequence
    # We need to track backpointers.
    # This complicates the DP significantly.
    # To get the actual sequence, we need to store prev_index and prev_gap for each state.
    
    # Let's restart with backpointers.
    # dp[i] = list of (gap, length, prev_idx, prev_gap)
    # Actually, we can just store the length and the state ID, or just reconstruct by searching.
    # Searching might be O(N) per step, so O(N^2) reconstruction.
    
    # Simpler approach:
    # Store dp[i] as list of (gap, length).
    # Also store prev[i][k] = index of previous prime for the k-th state in dp[i].
    # And prev_gap[i][k] = gap of the previous state.
    
    # Let's re-implement with this structure.
    
    if time.time() > deadline - 10.0:
         # If we ran out of time in the first pass, return a simple greedy result
         # to ensure we return *something* valid.
         # Greedy: start with 2, then pick smallest prime that maintains non-decreasing gap.
         seq = [2]
         last_gap = 0
         for p in primes[1:]:
             g = p - seq[-1]
             if g >= last_gap:
                 seq.append(p)
                 last_gap = g
         return seq

    dp = [[] for _ in range(n_primes)]
    # prev_idx[i][k] = index of previous prime for state k in dp[i]
    # prev_gap[i][k] = gap of the previous state (gap between prev_prime and prime_before_prev)
    # Actually, we need to know the gap of the *state* we are extending.
    # State (gap, len) in dp[i] means the last gap is `gap`.
    # To extend to j, we need gap_ij >= gap.
    # The new state in j will have gap_ij.
    # To backtrack, we need to know which state in i we came from.
    
    prev_idx = [[] for _ in range(n_primes)]
    prev_gap = [[] for _ in range(n_primes)]
    
    for j in range(n_primes):
        p_j = primes[j]
        candidates = []
        # Base case: starting new sequence
        candidates.append((0, 1, -1, 0)) # gap=0, len=1, prev=-1, prev_gap=0
        
        for i in range(j):
            gap_ij = p_j - primes[i]
            
            for k, (g_i, l_i) in enumerate(dp[i]):
                if g_i > gap_ij:
                    break
                new_len = l_i + 1
                # prev_idx for this new state in j is i
                # prev_gap for this new state is g_i (the gap of the state in i)
                # Wait, do we need to store prev_gap? 
                # When backtracking from j (state gap_ij), we go to i.
                # The state in i was (g_i, l_i).
                # So yes, we need to know which state k in i we came from.
                candidates.append((gap_ij, new_len, i, k))
        
        if time.time() > deadline - 5.0:
             break

        # Prune candidates
        # Key: (gap, -length)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        pruned = []
        pruned_prev_idx = []
        pruned_prev_gap = [] # Actually, we store the index k of the prev state, so we can look up its gap later?
        # No, let's just store the index k of the previous state.
        # Then we can look up dp[i][k] to get the gap and length.
        # So we only need to store (gap, length, prev_i, prev_k)
        
        pruned_states = []
        max_len_so_far = 0
        prev_gap_val = -1
        
        for gap, length, prev_i, prev_k in candidates:
            if gap == prev_gap_val:
                continue
            
            if length > max_len_so_far:
                pruned_states.append((gap, length, prev_i, prev_k))
                max_len_so_far = length
                prev_gap_val = gap
        
        dp[j] = pruned_states
        
    # Find best
    best_len = 0
    best_j = -1
    best_k = -1
    
    for j in range(n_primes):
        for k, state in enumerate(dp[j]):
            if state[1] > best_len:
                best_len = state[1]
                best_j = j
                best_k = k
    
    if best_j == -1:
        return [2]
        
    # Reconstruct
    seq = []
    curr_j = best_j
    curr_k = best_k
    
    while curr_j != -1:
        state = dp[curr_j][curr_k]
        seq.append(primes[curr_j])
        prev_i = state[2]
        prev_k = state[3]
        curr_j = prev_i
        curr_k = prev_k
        
    seq.reverse()
    
    # Verify validity (safety check)
    if not is_valid(seq):
        # Fallback
        return [2]
        
    return seq

# EVOLVE-BLOCK-END

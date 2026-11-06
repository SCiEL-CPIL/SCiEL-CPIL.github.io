import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Sieve of Eratosthenes to get primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    # dp[i] is a list of tuples (length, last_gap) representing non-dominated states
    # ending at primes[i]
    # A state (L1, G1) dominates (L2, G2) if L1 >= L2 and G1 <= G2
    dp = [None] * n
    
    # Initialize
    dp[0] = [(1, 0)]
    best_len = 1
    best_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        candidates = [(1, 0)]  # Subsequence consisting of only primes[i]
        
        for j in range(i):
            # Check if we can extend any state from j to i
            gap = primes[i] - primes[j]
            # We need a state (L, G) in dp[j] such that G <= gap
            # To be efficient, we can just iterate all states in dp[j]
            # Since the number of non-dominated states is small, this is fine
            for L, G in dp[j]:
                if G <= gap:
                    candidates.append((L + 1, gap))
        
        # Prune candidates to keep only non-dominated states
        # Sort by length descending, then by gap ascending
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        non_dom = []
        max_len_so_far = 0
        min_gap_so_far = float('inf')
        
        # We want to keep a state if it has a length greater than any previously kept state
        # OR if it has the same length but a smaller gap?
        # Actually, if we sort by -L, then G, the first one with a given L has the smallest G.
        # Any subsequent state with same L has larger G, so it is dominated.
        # Any subsequent state with smaller L is dominated if its G is larger than some kept state?
        # No. (L=5, G=10) and (L=4, G=1). Neither dominates the other.
        # So we need to keep all (L, G) such that there is no other (L', G') with L' >= L and G' <= G.
        
        # Efficient pruning:
        # Sort by Length descending.
        # Iterate through sorted list.
        # Keep a state (L, G) if G < min_gap_seen_so_far_for_lengths_ge_L?
        # Actually, since we process in descending L order:
        # If we have kept a state with L' >= L and G' <= G, then (L, G) is dominated.
        # So we just need to track the minimum gap seen so far in the sorted list.
        # If current G < min_gap_seen, it's not dominated by any previous (higher or equal length) state.
        # Wait, if L' > L and G' <= G, then dominated.
        # If L' == L and G' <= G, then dominated.
        # So yes, if we sort by -L, then G, we just need to ensure that for the current G,
        # it is strictly smaller than the minimum G encountered so far in the list of states with L' >= L.
        
        min_gap_encountered = float('inf')
        for L, G in candidates:
            if G < min_gap_encountered:
                non_dom.append((L, G))
                min_gap_encountered = G
        
        dp[i] = non_dom
        
        # Update global best
        # The best length ending at i is non_dom[0][0]
        current_best_len = non_dom[0][0]
        if current_best_len > best_len:
            best_len = current_best_len
            best_idx = i

    # Reconstruct the subsequence
    # We need to store parent pointers to reconstruct.
    # Let's redo the DP with parent pointers.
    
    # Restart DP with parents
    dp = [None] * n
    parent = [-1] * n # parent[i] = j, and we also need to know which state in dp[j] was used
    # Since we prune states, we need to be careful.
    # Let's store for each i, the best state (L, G) and the index j and the specific state in dp[j] that led to it.
    
    # To simplify reconstruction, we can just store the full path or backtrack.
    # Given the small N, we can store the previous index and the gap used.
    
    # Let's refine:
    # dp[i] = list of (L, G, prev_idx, prev_state_idx_in_dp_prev)
    # This might be heavy.
    # Alternative: Just find the length, then backtrack greedily?
    # No, we need the exact states.
    
    # Let's use a dictionary or list for states, and store indices.
    
    # For simplicity and speed, I will just return the length? No, task asks for the integers.
    # "Return increasing prime integers"
    
    # Let's do the DP again, storing the predecessor.
    
    # dp[i] is list of tuples: (length, last_gap, prev_prime_index, prev_state_index)
    # prev_state_index refers to the index of the state in dp[prev_prime_index]
    
    dp = [None] * n
    dp[0] = [(1, 0, -1, -1)]
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        candidates = [(1, 0, -1, -1)]
        
        for j in range(i):
            gap = primes[i] - primes[j]
            for k, (L, G, _, _) in enumerate(dp[j]):
                if G <= gap:
                    candidates.append((L + 1, gap, j, k))
        
        # Prune candidates
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        non_dom = []
        min_gap_encountered = float('inf')
        for cand in candidates:
            L, G, pi, si = cand
            if G < min_gap_encountered:
                non_dom.append(cand)
                min_gap_encountered = G
        
        dp[i] = non_dom
        
        current_best_len = non_dom[0][0]
        if current_best_len > best_len:
            best_len = current_best_len
            best_i = i
            best_state_idx = 0 # Index in dp[best_i]

    # Reconstruct
    seq = []
    i = best_i
    state_idx = best_state_idx
    while i != -1:
        L, G, prev_i, prev_state_idx = dp[i][state_idx]
        seq.append(primes[i])
        i = prev_i
        state_idx = prev_state_idx
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END

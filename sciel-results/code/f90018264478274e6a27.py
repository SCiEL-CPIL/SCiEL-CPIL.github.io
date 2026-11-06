import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N_LIMIT = 10000
    
    # 1. Sieve of Eratosthenes
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N_LIMIT) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n < 2:
        return primes
    
    # dp[i] is a list of non-dominated states for prime i.
    # State: (length, last_gap, prev_index)
    # Dominance: (l1, g1, p1) dominates (l2, g2, p2) if l1 >= l2 and g1 <= g2.
    # We sort by length descending, then gap ascending, and keep only the "frontier".
    
    dp = [[] for _ in range(n)]
    dp[0].append((1, 0, -1))
    
    best_len = 1
    best_idx = 0
    best_state = (1, 0, -1)
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_states = [(1, 0, -1)]
        
        pi_val = primes[i]
        
        for j in range(i):
            gap = pi_val - primes[j]
            
            # Iterate through states at j
            # Optimization: If we have a state with gap > gap, we can't extend it.
            # Since states are not sorted by gap in a simple way (sorted by len desc, gap asc),
            # we just check all.
            
            for (l, g, prev_idx) in dp[j]:
                if gap >= g:
                    new_states.append((l + 1, gap, j))
        
        # Sort and filter dominated states
        # Sort by length descending, then gap ascending
        new_states.sort(key=lambda x: (-x[0], x[1]))
        
        filtered = []
        # We keep a state if it is not dominated by any previously kept state.
        # Since we iterate by length descending, any state with length < current max length
        # and gap >= current min gap for that length is dominated?
        # Actually, dominance is: exists (l', g') in filtered such that l' >= l and g' <= g.
        # Because we process longer lengths first, if we have a state with l' >= l,
        # we check if its gap g' <= g.
        
        # To make this faster, we can maintain the minimum gap seen for each length?
        # Or simply, since the list is small, linear scan is fine.
        
        for s in new_states:
            l, g, prev_idx = s
            dominated = False
            for fs in filtered:
                fl, fg, _ = fs
                if fl >= l and fg <= g:
                    dominated = True
                    break
            if not dominated:
                filtered.append(s)
        
        dp[i] = filtered
        
        if filtered:
            # The first element in filtered has the max length (due to sort)
            # and min gap for that length (due to sort and filtering order? Not necessarily min gap for that length, but non-dominated).
            # Actually, the first element has the maximum length.
            # We want to maximize length.
            best_local = filtered[0]
            if best_local[0] > best_len:
                best_len = best_local[0]
                best_idx = i
                best_state = best_local
    
    # Reconstruct
    if best_len == 1:
        return [primes[best_idx]]
    
    path = []
    curr_idx = best_idx
    curr_len = best_state[0]
    curr_gap = best_state[1]
    curr_prev = best_state[2]
    
    # We need to trace back.
    # The state at curr_idx is (curr_len, curr_gap, curr_prev).
    # This state was formed by extending a state at curr_prev.
    # The state at curr_prev had length curr_len - 1.
    # And its gap g_prev must satisfy curr_gap >= g_prev.
    # However, there might be multiple states at curr_prev with length curr_len - 1.
    # We need the specific one that was used.
    # The current DP state only stores `prev_idx`. It does NOT store which specific state at `prev_idx` was used.
    # This is a flaw in the simple back-pointer approach if multiple states at `prev_idx` have the same length.
    # But wait, if we have two states at `prev_idx` with the same length `L-1`, say `(L-1, g1, p1)` and `(L-1, g2, p2)` with `g1 < g2`.
    # If we extend to `i` with gap `g_i >= g2`, then `g_i >= g1` is also true.
    # The state `(L, g_i, prev_idx)` is added.
    # When reconstructing, if we just look for a state at `prev_idx` with length `L-1`, we might pick the wrong one?
    # No, the gap constraint is `g_i >= g_prev`.
    # If we pick a state with `g_prev` such that `g_i < g_prev`, it's invalid.
    # But we know the transition was valid.
    # The issue is: which state at `prev_idx` was the one that led to `curr_idx`?
    # If we have multiple non-dominated states at `prev_idx` with the same length, they have different gaps.
    # The one used must have gap <= curr_gap.
    # If there are multiple, does it matter?
    # The previous gap `g_prev` becomes the "last gap" for the previous step.
    # For the next step back, we need to know `g_prev` to check against the gap before that.
    # So we MUST know the specific state.
    
    # To fix this, we should store the previous gap in the state, or the index of the state in dp[prev_idx].
    # Let's modify the state to include the index of the previous state in dp[prev_idx].
    # State: (length, last_gap, prev_prime_idx, prev_state_idx)
    
    # Let's re-run the DP with this correction.
    
    dp = [[] for _ in range(n)]
    dp[0].append((1, 0, -1, -1)) # (len, gap, prime_idx, state_idx_in_dp)
    
    best_len = 1
    best_idx = 0
    best_state = (1, 0, -1, -1)
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_states = [(1, 0, -1, -1)]
        
        pi_val = primes[i]
        
        for j in range(i):
            gap = pi_val - primes[j]
            
            for k, (l, g, p_idx, s_idx) in enumerate(dp[j]):
                if gap >= g:
                    new_states.append((l + 1, gap, j, k))
        
        new_states.sort(key=lambda x: (-x[0], x[1]))
        
        filtered = []
        for s in new_states:
            l, g, p_idx, s_idx = s
            dominated = False
            for fs in filtered:
                fl, fg, _, _ = fs
                if fl >= l and fg <= g:
                    dominated = True
                    break
            if not dominated:
                filtered.append(s)
        
        dp[i] = filtered
        
        if filtered:
            best_local = filtered[0]
            if best_local[0] > best_len:
                best_len = best_local[0]
                best_idx = i
                best_state = best_local
    
    # Reconstruct
    if best_len == 1:
        return [primes[best_idx]]
    
    path = []
    curr_idx = best_idx
    curr_state_idx = best_state[3]
    
    while curr_idx != -1:
        path.append(primes[curr_idx])
        if curr_state_idx == -1:
            break
        
        # Get the state
        l, g, prev_prime_idx, prev_state_idx = dp[curr_idx][curr_state_idx]
        curr_idx = prev_prime_idx
        curr_state_idx = prev_state_idx
    
    path.reverse()
    return path

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
    
    # DP: For each prime index i, maintain a list of states (length, last_gap, prev_index)
    # where the sequence ends at primes[i]
    # A state (l, g, p) is dominated by (l', g', p') if l' >= l and g' <= g
    # We keep only non-dominated states
    
    dp = [[] for _ in range(n)]
    dp[0].append((1, 0, -1))
    
    best_len = 1
    best_idx = 0
    best_state = (1, 0, -1)
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_states = [(1, 0, -1)]  # Base: sequence starting at i
        
        for j in range(i):
            gap = primes[i] - primes[j]
            for (l, g, pi) in dp[j]:
                if gap >= g:
                    new_states.append((l + 1, gap, j))
        
        # Sort by length descending, then gap ascending for dominance check
        new_states.sort(key=lambda x: (-x[0], x[1]))
        
        # Filter dominated states
        filtered = []
        for s in new_states:
            l, g, pi = s
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
            best_local = filtered[0]
            if best_local[0] > best_len:
                best_len = best_local[0]
                best_idx = i
                best_state = best_local
    
    # Reconstruct sequence
    if best_len == 1:
        return [primes[best_idx]]
    
    path = []
    curr_idx = best_idx
    curr_len = best_state[0]
    curr_gap = best_state[1]
    curr_prev = best_state[2]
    
    while curr_idx != -1:
        path.append(primes[curr_idx])
        if curr_len == 1:
            break
        
        # Find the state in dp[curr_idx] that matches (curr_len, curr_gap)
        target_prev = -1
        for (l, g, pi) in dp[curr_idx]:
            if l == curr_len and g == curr_gap:
                target_prev = pi
                break
        
        if target_prev == -1:
            break
        
        # Now we need to find the state at target_prev that was used
        # The gap from target_prev to curr_idx is curr_gap
        # We need the state at target_prev with length curr_len-1
        # and the gap that leads to curr_gap
        # Actually, the state at target_prev has some gap g_prev, and curr_gap >= g_prev
        # We need to find which state at target_prev was extended
        
        # The state at target_prev that was used must have length curr_len - 1
        # and its gap g_prev must satisfy curr_gap >= g_prev
        # There could be multiple such states, but we need the specific one
        
        # Better approach: store the full path or use a different reconstruction
        # Let's store prev pointers more carefully
        
        curr_idx = target_prev
        curr_len -= 1
        # We lost track of the gap at the previous step
        # This reconstruction is broken
    
    # The reconstruction above is flawed. Let me use a different approach:
    # Store the actual sequence indices in each state
    
    # Redo with proper reconstruction
    dp2 = [[] for _ in range(n)]
    dp2[0].append((1, 0, -1, [0]))
    
    best_len2 = 1
    best_idx2 = 0
    best_state2 = (1, 0, -1, [0])
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        new_states = [(1, 0, -1, [i])]
        
        for j in range(i):
            gap = primes[i] - primes[j]
            for (l, g, pi, seq) in dp2[j]:
                if gap >= g:
                    new_states.append((l + 1, gap, j, seq + [i]))
        
        new_states.sort(key=lambda x: (-x[0], x[1]))
        
        filtered2 = []
        for s in new_states:
            l, g, pi, seq = s
            dominated = False
            for fs in filtered2:
                fl, fg, _, _ = fs
                if fl >= l and fg <= g:
                    dominated = True
                    break
            if not dominated:
                filtered2.append(s)
        
        dp2[i] = filtered2
        
        if filtered2:
            best_local2 = filtered2[0]
            if best_local2[0] > best_len2:
                best_len2 = best_local2[0]
                best_idx2 = i
                best_state2 = best_local2
    
    if best_len2 == 1:
        return [primes[best_idx2]]
    
    final_seq_indices = best_state2[3]
    return [primes[i] for i in final_seq_indices]

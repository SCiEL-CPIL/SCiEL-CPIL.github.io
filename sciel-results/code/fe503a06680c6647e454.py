"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    n = len(primes)
    
    if n == 0:
        return []
        
    # dp[i] is a list of tuples (gap, length, prev_idx, prev_state_idx)
    dp = [None] * n
    # State for the first prime: gap 0, length 1, no previous
    dp[0] = [(0, 1, -1, -1)]
    
    best_len = 1
    best_idx = 0
    best_state_idx = 0
    
    for j in range(1, n):
        if time.time() > deadline:
            break
            
        new_states = []
        pj = primes[j]
        
        # Always can start a new sequence at j
        new_states.append((0, 1, -1, -1))
        
        for i in range(j):
            pi = primes[i]
            gap_ij = pj - pi
            # Iterate over states in dp[i]
            for state_idx, (g, L, p_idx, p_s_idx) in enumerate(dp[i]):
                if gap_ij >= g:
                    new_L = L + 1
                    new_states.append((gap_ij, new_L, i, state_idx))
                    
        # Prune to Pareto frontier
        # Sort by length descending, then gap ascending
        new_states.sort(key=lambda x: (-x[1], x[0]))
        
        frontier = []
        min_gap = float('inf')
        for state in new_states:
            g, L, p_idx, p_s_idx = state
            if g < min_gap:
                frontier.append(state)
                min_gap = g
        
        dp[j] = frontier
        
        # Update global best
        # The first state in frontier has the maximum length
        if frontier:
            max_L = frontier[0][1]
            if max_L > best_len:
                best_len = max_L
                best_idx = j
                best_state_idx = 0
            elif max_L == best_len:
                # If lengths are equal, prefer the one with smaller gap for potential future extensions?
                # Or just keep the first one found.
                # Actually, for the final answer, any valid sequence of max length works.
                # But having a smaller gap at the end doesn't help since we are done.
                # However, if we found a tie earlier, we might want the one with smaller gap
                # if we were to continue? But we are not.
                # So we can just keep the first one.
                pass

    # Reconstruct sequence
    seq = []
    curr_idx = best_idx
    curr_state_idx = best_state_idx
    
    while curr_idx != -1:
        state = dp[curr_idx][curr_state_idx]
        g, L, p_idx, p_s_idx = state
        seq.append(primes[curr_idx])
        curr_idx = p_idx
        curr_state_idx = p_s_idx
        
    seq.reverse()
    return seq
# EVOLVE-BLOCK-END

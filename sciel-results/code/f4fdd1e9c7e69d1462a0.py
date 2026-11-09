import time

def construct():
    start_time = time.time()
    deadline = 90.0
    
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    dp = [None] * n
    best_state = [None] * n
    
    dp[0] = [(0, 1, -1, -1)]
    best_state[0] = (0, 1, -1, -1)
    max_len = 1
    best_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline - 10.0:
            break
            
        candidates = []
        pi = primes[i]
        
        for j in range(i):
            gap = pi - primes[j]
            for prev_gap, prev_len, prev_idx, prev_prev_gap in dp[j]:
                if gap >= prev_gap:
                    candidates.append((gap, prev_len + 1, j, prev_gap))
        
        if not candidates:
            dp[i] = []
            best_state[i] = None
            continue

        # Sort by gap, then by length descending
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        pruned = []
        max_len_so_far = 0
        for gap, length, prev_idx, prev_gap in candidates:
            if length > max_len_so_far:
                pruned.append((gap, length, prev_idx, prev_gap))
                max_len_so_far = length
        
        dp[i] = pruned
        
        if pruned:
            curr_best = pruned[-1]
            best_state[i] = curr_best
            if curr_best[1] > max_len:
                max_len = curr_best[1]
                best_idx = i

    # Reconstruct path
    path = []
    curr_idx = best_idx
    curr_gap = best_state[best_idx][0] if best_state[best_idx] else 0
    
    while curr_idx != -1:
        path.append(primes[curr_idx])
        state = best_state[curr_idx]
        if state is None:
            break
        prev_idx = state[2]
        prev_gap = state[3]
        
        if prev_idx == -1:
            curr_idx = -1
        else:
            curr_idx = prev_idx
            # We don't strictly need curr_gap for the next step's state lookup 
            # because best_state[prev_idx] is the specific state we came from?
            # NO. best_state[prev_idx] is the state in dp[prev_idx] with MAX LENGTH.
            # The state we came from might NOT be the max length state of dp[prev_idx].
            # We need to find the state in dp[prev_idx] with gap == prev_gap.
            
            # Optimization: dp[prev_idx] is sorted by gap. We can binary search or linear scan.
            # Given small size, linear scan is fine.
            target_state = None
            for s in dp[prev_idx]:
                if s[0] == prev_gap:
                    target_state = s
                    break
            
            if target_state is None:
                # Error in logic
                break
            else:
                # Update best_state? No, best_state is only for the final path.
                # We just need to move to the correct state.
                # The loop continues, and we will look up the parent of target_state.
                # So we need to set curr_idx and the gap to match target_state.
                # But the loop condition uses best_state[curr_idx].
                # This structure is flawed for backtracking non-max states.
                
                # Fix: Store the full path in the state? No.
                # Fix: During backtracking, don't rely on best_state array.
                # Instead, keep the current state tuple.
                pass

    # Correct Backtracking Logic
    path = []
    curr_idx = best_idx
    # Get the specific state we want to start from
    # It is the state in dp[best_idx] with max length.
    # Which is best_state[best_idx].
    curr_state = best_state[best_idx]
    
    while curr_idx != -1:
        path.append(primes[curr_idx])
        if curr_state is None:
            break
        prev_idx = curr_state[2]
        prev_gap = curr_state[3]
        
        if prev_idx == -1:
            break
        
        # Find the state in dp[prev_idx] with gap == prev_gap
        next_state = None
        for s in dp[prev_idx]:
            if s[0] == prev_gap:
                next_state = s
                break
        
        if next_state is None:
            break
            
        curr_idx = prev_idx
        curr_state = next_state
        
    path.reverse()
    return path

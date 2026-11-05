import time
from array import array

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin for 90s limit
    
    limit = 10000
    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    M = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    G = limit  # Max possible gap
    
    # best[i][g] = max length of subsequence ending at i with last gap <= g
    # best_arg[i][g] = the gap g' <= g that achieves best[i][g]
    # prev_idx[i][g] = the index j of the previous prime in the sequence for state (i, g)
    # prev_gap[i][g] = the gap g_prev used at j for state (i, g)
    
    # We only need to store best and best_arg for all i, to be used by future i's.
    # We don't need to store dp explicitly, just best.
    # But for backtracking, we need prev_idx and prev_gap for the specific path.
    # We can store prev_idx and prev_gap for all states? That's 2 * 12M * 2 bytes = 48MB. Acceptable.
    # Or we can store them only if they are part of a potential max?
    # Let's store them for all states to make backtracking simple.
    
    # Use array('H') for 16-bit unsigned integers.
    # 0 is used for "no previous" or base case.
    
    best = [array('H', [0]) * (G + 1) for _ in range(M)]
    best_arg = [array('H', [0]) * (G + 1) for _ in range(M)]
    prev_idx = [array('H', [0]) * (G + 1) for _ in range(M)]
    prev_gap = [array('H', [0]) * (G + 1) for _ in range(M)]
    
    # Mark 0 as invalid index for prev_idx? 
    # If prev_idx[i][g] == 0, it could mean index 0 or no previous.
    # Let's use 0 to mean "no previous" (start of sequence).
    # But index 0 is a valid prime.
    # Let's use 65535 (max H) to mean "no previous".
    # Or just handle the base case separately.
    # Base case: dp[i][g] = 1. This means the sequence is just [primes[i]].
    # In this case, there is no previous prime.
    
    max_len = 0
    max_i = -1
    max_g = -1
    
    for i in range(M):
        if time.time() > deadline - 5:
            break
            
        p_i = primes[i]
        best_i = array('H', [0]) * (G + 1)
        best_arg_i = array('H', [0]) * (G + 1)
        prev_idx_i = array('H', [0]) * (G + 1)
        prev_gap_i = array('H', [0]) * (G + 1)
        
        current_max = 0
        current_arg = 0
        
        # dp_i[g] is not explicitly stored, we compute best_i directly.
        # best_i[g] = max(best_i[g-1], dp_i[g])
        # dp_i[g] = 1 + best[j][g] if p_i - g is prime p_j, else 1.
        
        for g in range(1, G + 1):
            # Calculate dp_i[g]
            p_prev = p_i - g
            dp_val = 1
            prev_i_val = 0  # 0 means no previous
            prev_g_val = 0
            
            if p_prev in prime_to_idx:
                j = prime_to_idx[p_prev]
                # best[j][g] is the max length ending at j with last gap <= g
                # We add 1 for the new prime i
                candidate_len = 1 + best[j][g]
                if candidate_len > dp_val:
                    dp_val = candidate_len
                    prev_i_val = j
                    # The gap at j that was used is best_arg[j][g]
                    prev_g_val = best_arg[j][g]
            
            # Update best_i
            if dp_val > current_max:
                current_max = dp_val
                current_arg = g
                # Update tracking of global max
                if current_max > max_len:
                    max_len = current_max
                    max_i = i
                    max_g = g
            
            best_i[g] = current_max
            best_arg_i[g] = current_arg
            prev_idx_i[g] = prev_i_val if dp_val == current_max else 0 # This logic is flawed.
            # We need prev_idx_i[g] to correspond to the state (i, g) specifically, 
            # not the best state up to g.
            # But best_i[g] is the max up to g.
            # For backtracking, we follow the path of max_len.
            # The state (i, g) has value dp_i[g].
            # If dp_i[g] == best_i[g], then the path to (i, g) is the best path to i with gap <= g.
            # If dp_i[g] < best_i[g], then the best path to i with gap <= g does NOT end with gap g.
            # So we only need prev_idx for states that are "active" in the max.
            # Actually, we can just store prev_idx_i[g] and prev_gap_i[g] for the specific dp_i[g].
            # And when backtracking, we know we are at (i, g).
            # We look at prev_idx_i[g] and prev_gap_i[g].
            # If prev_idx_i[g] is 0, we stop.
            # But wait, if dp_i[g] < best_i[g], then (i, g) is not the end of the best sequence ending at i.
            # However, we are backtracking from the GLOBAL max.
            # The global max is at (max_i, max_g).
            # dp[max_i][max_g] == best[max_i][max_g] by definition of max_g?
            # Not necessarily. max_g is the g that achieved the global max.
            # So dp[max_i][max_g] is the global max.
            # So we just need to store prev_idx and prev_gap for ALL (i, g).
            
            prev_idx_i[g] = prev_i_val
            prev_gap_i[g] = prev_g_val
        
        best[i] = best_i
        best_arg[i] = best_arg_i
        prev_idx[i] = prev_idx_i
        prev_gap[i] = prev_gap_i

    if max_len == 0:
        return [2]
        
    # Backtrack
    path = []
    i = max_i
    g = max_g
    while i != -1:
        path.append(primes[i])
        j = prev_idx[i][g]
        if j == 0:
            # Check if j==0 means "no previous" or "index 0".
            # If i > 0 and g > 0, and p_i - g is not prime, then j=0.
            # If i == 0, j=0.
            # How to distinguish?
            # If j == 0, it means either no previous or previous is index 0.
            # If previous is index 0, then p_i - g == primes[0] = 2.
            # We can check if p_i - g == 2.
            if i > 0 and primes[i] - g == 2:
                # Previous is index 0
                pass # j is indeed 0
            else:
                break # No previous
        # The previous state was (j, prev_gap[i][g])
        g = prev_gap[i][g]
        i = j
        
    path.reverse()
    return path

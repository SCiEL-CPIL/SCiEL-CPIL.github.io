import time
import sys

def construct():
    start_time = time.time()
    # 90s limit. Use 80s for safety.
    deadline = start_time + 80.0
    
    N = 256
    order = list(range(1, N + 1))
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    # Precompute suffix sums of weights for the loose upper bound
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order[i]]
        
    best_score = 0.0
    best_mask = 0  # Bitset representation of the best set found
    
    # Helper to check if x can be added to the set represented by mask
    def is_valid_add(mask, x):
        # Iterate through set bits in mask
        # mask has bits 0..255 corresponding to 1..256
        m = mask
        while m:
            # Get index of lowest set bit
            # lsb = m & -m
            # idx = lsb.bit_length() - 1
            # a = idx + 1
            # This is slow in Python loop. 
            # Alternative: use a list of active elements? 
            # But mask is passed by value (int), so we don't have a list.
            # We can maintain a list of elements in the set alongside the mask?
            # Or just convert mask to list? No, too slow.
            # Let's stick to bit iteration but optimize.
            lsb = m & -m
            idx = lsb.bit_length() - 1
            a = idx + 1
            
            # Check 1: x is middle. a + c = 2x => c = 2x - a
            c = 2 * x - a
            if 1 <= c <= N:
                if mask & (1 << (c - 1)):
                    return False
            
            # Check 2: a is middle. x + c = 2a => c = 2a - x
            c2 = 2 * a - x
            if 1 <= c2 <= N:
                if mask & (1 << (c2 - 1)):
                    return False
            
            # Check 3: x is endpoint, a is endpoint? 
            # Midpoint m = (x+a)/2. If m is in S, then {a, m, x} is AP.
            s = a + x
            if s % 2 == 0:
                mid = s // 2
                if 1 <= mid <= N:
                    if mask & (1 << (mid - 1)):
                        return False
            
            m ^= lsb
        return True

    def solve(idx, mask, current_sum):
        nonlocal best_score, best_mask
        
        if time.time() > deadline:
            return
            
        # Loose Pruning
        if current_sum + suffix_sum[idx] <= best_score:
            return
            
        # Tight Pruning (only if loose bound is promising)
        # Calculate sum of weights of remaining elements that are currently valid to add
        # Note: This is an upper bound because adding one might make another invalid.
        # But it's tighter than suffix_sum.
        tight_ub = current_sum
        # We can stop early if tight_ub <= best_score
        # Optimization: Precompute a list of (index, weight, value) for remaining?
        # Just loop.
        for i in range(idx, N):
            x = order[i]
            # Quick check: if x is already in mask? No, idx ensures x > all in mask? 
            # No, mask contains elements from 1..idx-1. x is idx+1..N.
            # So x is not in mask.
            if is_valid_add(mask, x):
                tight_ub += weights[x]
                if tight_ub > best_score + 1e-9: # Small epsilon
                    # We might still be able to improve, but we can't prune yet
                    # We need the full tight_ub to compare? 
                    # Actually, if we just want to prune, we can break if we know 
                    # even adding all remaining valid ones won't beat best?
                    # No, we are summing them up. 
                    # We can't break early unless we have a lower bound on the rest?
                    # The rest are 0. So tight_ub is the sum of all valid ones.
                    # We must finish the loop to get the true tight_ub.
                    pass
        
        if tight_ub <= best_score + 1e-12:
            return
            
        if idx == N:
            if current_sum > best_score:
                best_score = current_sum
                best_mask = mask
            return
            
        x = order[idx]
        
        # Option 1: Include x
        if is_valid_add(mask, x):
            new_mask = mask | (1 << (x - 1))
            solve(idx + 1, new_mask, current_sum + weights[x])
            
        # Option 2: Exclude x
        solve(idx + 1, mask, current_sum)

    # Start B&B
    solve(0, 0, 0.0)
    
    # Local Search (Hill Climbing)
    # Convert best_mask to set
    S = set()
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        S.add(idx + 1)
        m ^= lsb
        
    def get_set_score(S):
        return sum(1.0 / x for x in S)
    
    def is_valid_set(S):
        # Check if S is 3-AP free
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                s = a + b
                if s % 2 == 0:
                    mid = s // 2
                    if mid in S:
                        return False
        return True

    # Swap-based local search
    # Try to replace i in S with j not in S (j < i) to increase weight
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Sort S in descending order to try removing larger (lower weight) elements first?
        # Or sort candidates j in ascending order (higher weight) first.
        candidates_in = sorted([x for x in range(1, N + 1) if x not in S])
        # We only care about j < some i in S.
        # For each j in candidates_in (ascending weight):
        for j in candidates_in:
            if time.time() > deadline - 2.0:
                break
            # Try to find an i in S such that i > j and (S - {i} + {j}) is valid
            # And 1/j > 1/i (which is true if j < i)
            S_list = sorted(S, reverse=True) # Try removing largest i first (lowest weight)
            for i in S_list:
                if i <= j:
                    break # Since S_list is desc, if i <= j, all subsequent are <= j.
                # Check if swapping i with j is valid
                S_temp = S.copy()
                S_temp.remove(i)
                S_temp.add(j)
                # Check validity of S_temp
                # We can do a fast check: only need to check APs involving j.
                # j is new. So check if j forms AP with any pair in S_temp \ {j}.
                # Pairs in S_temp \ {j} are from S \ {i}.
                valid = True
                for a in S_temp:
                    if a == j: continue
                    c = 2 * j - a
                    if c in S_temp:
                        valid = False
                        break
                    c2 = 2 * a - j
                    if c2 in S_temp:
                        valid = False
                        break
                    s = a + j
                    if s % 2 == 0:
                        mid = s // 2
                        if mid in S_temp:
                            valid = False
                            break
                if valid:
                    # Perform swap
                    S = S_temp
                    improved = True
                    # Restart the loop to find more improvements
                    break
            if improved:
                break

    return list(S)

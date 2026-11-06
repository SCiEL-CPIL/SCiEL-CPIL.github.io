def construct():
    import time
    from math import gcd

    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe

    N = 256
    S = list(range(2, 129))  # Small numbers
    L = list(range(129, N + 1))  # Large numbers
    
    nS = len(S)
    nL = len(L)
    
    # Map value to index in L for bitset operations
    val_to_l_idx = {v: i for i, v in enumerate(L)}
    
    # Precompute reciprocal weights
    w_S = {v: 1.0 / v for v in S}
    w_L = [1.0 / v for v in L]
    
    # Precompute conflict bitsets C_k for k in S
    # C_k is an integer bitmask where bit i is 1 if lcm(k, L[i]) <= 256
    C = {}
    for k in S:
        mask = 0
        for i, l in enumerate(L):
            # lcm(k, l) = k * l // gcd(k, l)
            if k * l // gcd(k, l) <= N:
                mask |= (1 << i)
        C[k] = mask
        
    # Precompute sum of weights for L subsets? 
    # No, we will calculate marginal penalty on the fly using bit operations.
    # To speed up summing weights of bits in a mask, we can precompute 
    # a lookup table for small chunks or just iterate bits.
    # Given nL=128, iterating bits is fast enough if masks are sparse.
    
    def mask_penalty(mask):
        """Calculate sum of w_L[i] for i in mask."""
        s = 0.0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += w_L[idx]
            m ^= lsb
        return s

    # Greedy initialization
    # Sort S by 1/k descending
    S_sorted = sorted(S, key=lambda x: -w_S[x])
    best_K = []
    best_U = 0
    best_score = 0.0
    
    # Simple greedy: add k if independent in S and net gain > 0
    # Independence in S: check lcm(k, existing) > 256
    current_K = []
    current_U = 0
    current_penalty = 0.0
    current_gain = 0.0
    
    for k in S_sorted:
        # Check conflict with current_K in S
        ok = True
        for existing in current_K:
            if k * existing // gcd(k, existing) <= N:
                ok = False
                break
        if not ok:
            continue
        
        # Calculate marginal penalty
        new_mask = C[k] & ~current_U
        marginal_penalty = mask_penalty(new_mask)
        
        net_gain = w_S[k] - marginal_penalty
        if net_gain > 0:
            current_K.append(k)
            current_U |= C[k]
            current_penalty += marginal_penalty
            current_gain += w_S[k]
            
    if current_gain - current_penalty > best_score:
        best_score = current_gain - current_penalty
        best_K = current_K[:]
        best_U = current_U
        
    # Branch and Bound
    # We search for a better set.
    # We will use a recursive function with a node counter/time check.
    
    # Order S for B&B: by 1/k descending
    # We need to track which nodes are available.
    # Since we already have a good greedy solution, we can prune aggressively.
    
    def backtrack(idx, current_K, current_U, current_gain, current_penalty, remaining_sum):
        nonlocal best_score, best_K, best_U
        if time.time() > deadline:
            return
        
        # Upper Bound
        # Max additional gain is sum of 1/k for remaining k in S_sorted[idx:]
        # Min additional penalty is 0
        ub = current_gain + remaining_sum - current_penalty
        
        if ub <= best_score:
            return
            
        if idx == nS:
            if current_gain - current_penalty > best_score:
                best_score = current_gain - current_penalty
                best_K = current_K[:]
                best_U = current_U
            return
        
        k = S_sorted[idx]
        
        # Option 1: Skip k
        backtrack(idx + 1, current_K, current_U, current_gain, current_penalty, remaining_sum - w_S[k])
        
        # Option 2: Include k
        # Check conflict with current_K
        ok = True
        for existing in current_K:
            if k * existing // gcd(k, existing) <= N:
                ok = False
                break
        
        if ok:
            new_mask = C[k] & ~current_U
            marginal_penalty = mask_penalty(new_mask)
            new_gain = current_gain + w_S[k]
            new_penalty = current_penalty + marginal_penalty
            new_U = current_U | C[k]
            
            # Prune if new score + remaining sum (minus k) <= best
            # Note: remaining_sum in recursive call will be remaining_sum - w_S[k]
            # So we can just call and let the UB check handle it, 
            # but we can pre-check:
            if new_gain - new_penalty + (remaining_sum - w_S[k]) > best_score:
                current_K.append(k)
                backtrack(idx + 1, current_K, new_U, new_gain, new_penalty, remaining_sum - w_S[k])
                current_K.pop()

    # Initial remaining sum
    total_S_sum = sum(w_S.values())
    
    # To make B&B faster, we can restrict the search space.
    # The greedy solution is likely close to optimal.
    # We can also limit the depth or branching factor if time is tight, 
    # but the UB should prune most branches.
    
    # One issue: Python recursion limit. nS=127 is fine.
    # But the number of nodes might be large.
    # We can add a node limit or just rely on time.
    
    # Let's run B&B
    backtrack(0, [], 0, 0.0, 0.0, total_S_sum)
    
    # Construct result
    # best_K is the set of small numbers
    # We need to add all l in L that are compatible with best_K
    # l is compatible if not in best_U
    
    result = list(best_K)
    for i, l in enumerate(L):
        if not (best_U & (1 << i)):
            result.append(l)
            
    if not result:
        result = [2]
        
    return sorted(result)

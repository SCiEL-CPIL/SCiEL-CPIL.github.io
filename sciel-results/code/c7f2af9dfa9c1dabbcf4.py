import time
from math import gcd
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # 1. Calculate LCM of 2..48
    L = 1
    for i in range(2, 49):
        L = L * i // gcd(L, i)
        
    # 2. Weights
    # Indices 0..46 correspond to denominators 2..48
    weights = [L // (i + 2) for i in range(47)]
    
    total_weight = sum(weights)
    target = total_weight - L
    
    # We want subset R such that sum(R) > target, minimizing sum(R).
    
    # Split
    mid = 23
    weights_A = weights[:mid]
    weights_B = weights[mid:]
    
    def gen_sums_masks(weights_list):
        results = [(0, 0)]
        n = len(weights_list)
        for k in range(n):
            w = weights_list[k]
            bit = 1 << k
            new_results = []
            # Optimization: if time is tight, we might not generate all, but 2^23 is small enough
            for s, m in results:
                ns = s + w
                new_results.append((ns, m | bit))
            results.extend(new_results)
        return results

    # Generate sums for A
    results_A = gen_sums_masks(weights_A)
    if time.time() > deadline:
        return [0] # Fallback
        
    # Generate sums for B
    results_B = gen_sums_masks(weights_B)
    if time.time() > deadline:
        return [0] # Fallback

    # Sort B by sum
    results_B.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums_B = [x[0] for x in results_B]
    masks_B = [x[1] for x in results_B]
    
    best_delta = None
    best_mask_A = 0
    best_mask_B = 0
    
    # Iterate through A
    # For each s_A, we need s_B such that s_A + s_B > target
    # => s_B > target - s_A
    # => s_B >= target - s_A + 1
    
    # We can sort A as well? Not necessary, but might help if we prune.
    # Let's just iterate.
    
    for s_A, m_A in results_A:
        # If s_A alone is already > target, we need smallest s_B (which is 0? No, R must be non-empty? 
        # Wait, S must be non-empty. R is complement.
        # If R is empty, S is full set. Sum(S) = total_weight/L > 1. Invalid.
        # So S cannot be the full set.
        # This implies R cannot be empty.
        # So sum(R) must be > 0.
        # My condition sum(R) > target ensures sum(R) is large.
        # Is it possible that the optimal R is empty? No, because sum(empty)=0, and target is huge.
        # So R will never be empty in the optimal solution for "minimize sum > target" where target is large.
        
        req = target - s_A + 1
        if req < 0:
            req = 0 # We need s_B >= 0. Smallest s_B is 0.
            
        # Find first index in sums_B >= req
        idx = bisect_left(sums_B, req)
        
        if idx < len(sums_B):
            s_B = sums_B[idx]
            total_R = s_A + s_B
            # Check if total_R > target (it should be by construction)
            if total_R > target:
                delta = total_R - target
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_mask_A = m_A
                    best_mask_B = masks_B[idx]
                    
                    # Early exit if delta == 1? 
                    # Since weights are integers, minimal positive deficit numerator is 1.
                    if best_delta == 1:
                        break
                        
    # Construct indices for S (complement of R)
    # R mask is best_mask_A | (best_mask_B << mid)
    # S is all indices NOT in R.
    
    # Wait, the problem asks for indices into denominators [2..48].
    # If R is the set of excluded items, S is the set of included items.
    # We want to return indices of S.
    
    indices = []
    for i in range(47):
        if i < mid:
            bit = 1 << i
            if not (best_mask_A & bit):
                indices.append(i)
        else:
            bit = 1 << (i - mid)
            if not (best_mask_B & bit):
                indices.append(i)
                
    if not indices:
        # Fallback if something went wrong (e.g., R was full set, S empty)
        # But R sum > target ~ 3L, so R is large. S is small.
        # If S is empty, sum is 0. Deficit 1.
        # But we found a valid R, so S should be non-empty unless R is all indices.
        # If R is all indices, sum(R) = total_weight.
        # Is total_weight > target? Yes.
        # Is it the minimum? Unlikely, we can remove some weights.
        # So S should be non-empty.
        indices = [0]
        
    return sorted(indices)

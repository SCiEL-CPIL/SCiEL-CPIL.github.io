import numpy as np

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    MAX_L = 5000  # Increase search horizon slightly to allow for better optimization
    x = np.arange(1, MAX_L + 1)
    
    coverage = np.zeros(MAX_L, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    def compute_L(cov):
        # Find largest L such that all x in [1,L] have coverage >= 2
        # We want the first index where cov < 2.
        # cov is 0-indexed, x is 1-indexed.
        # If cov[0] < 2, L=0.
        # If cov[0]>=2, cov[1]<2, L=1.
        # ...
        # If all cov[i] >= 2, L=MAX_L.
        
        # Vectorized approach:
        # bad = cov < 2
        # If any bad, L = index of first bad.
        # Else L = MAX_L.
        
        bad = np.where(cov < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0]

    def get_deficit_score(r, p, current_cov, current_L):
        """
        Score a residue choice based on how much it helps the prefix.
        We prioritize extending L and increasing coverage near the bottleneck.
        """
        # We simulate adding this prime.
        # new_cov = current_cov + (x % p == r)
        
        # We don't want to copy the whole array.
        # We check the prefix [0, current_L + margin]
        
        # Actually, let's just compute the new L and a "density" score.
        # The most important metric is the new L.
        
        # Optimization: 
        # If current_cov[i] >= 2, adding 1 doesn't change "safety" (still >=2).
        # If current_cov[i] == 1, adding 1 makes it 2 (safe).
        # If current_cov[i] == 0, adding 1 makes it 1 (unsafe).
        
        # So, L is limited by the first i where current_cov[i] == 0.
        # UNLESS r covers that i. If r covers it, current_cov[i] becomes 1, still unsafe.
        # Wait. 
        # If current_cov[i] == 0, new_cov[i] is either 0 (if not covered) or 1 (if covered).
        # In both cases, new_cov[i] < 2.
        # So if there is any i with current_cov[i] == 0, L cannot pass i.
        
        # Therefore, the bottleneck is the first i with current_cov[i] == 0.
        # Let's call this idx_zero.
        # If no idx_zero, then L is determined by the first i where current_cov[i] == 1 AND (x % p != r).
        
        idx_zero = np.where(current_cov == 0)[0]
        if len(idx_zero) > 0:
            first_zero = idx_zero[0]
            # L is at most first_zero.
            # Actually, if current_cov[first_zero] == 0, then new_cov[first_zero] <= 1.
            # So L = first_zero.
            # This is independent of r?
            # Yes, because 0+1=1 < 2.
            # So if there is a 0-coverage point, we cannot pass it with just one more prime.
            # This implies that to extend L, we must have NO 0-coverage points in the prefix.
            
            # This suggests the greedy strategy must ensure we never have a 0-coverage point?
            # No, we have 168 primes. 
            # Initially, coverage is all 0.
            # First prime p=2. r=0 or 1.
            # If r=0, x=2,4,6... covered. x=1,3,5... not.
            # So coverage has 0s.
            # L=0? 
            # Task says "largest L such that every integer 1..L satisfies at least two".
            # If L=0, it's trivially true?
            # "every integer 1..0" is empty set. True.
            # But we want to maximize L.
            
            # So the logic "L is bounded by first zero" is correct for a SINGLE addition.
            # But we are adding sequentially.
            # After adding prime 1, we have 0s and 1s.
            # After adding prime 2, we can turn some 0s into 1s, and some 1s into 2s.
            # We cannot turn a 0 into a 2.
            # So to get L > 0, we need x=1 to have coverage >= 2.
            # This requires at least 2 primes to cover x=1.
            
            # So the "first zero" logic applies to the *current* state.
            # If current_cov has a 0 at index k, then new_cov will have < 2 at k.
            # So L_new <= k.
            
            # To maximize L, we should try to eliminate 0s?
            # No, we can't eliminate 0s with one prime. We can only make them 1s.
            # We need to make them 2s. That requires 2 primes to cover them.
            
            # So, the strategy should be:
            # 1. Identify the first index k where current_cov[k] == 0.
            # 2. Any residue r that covers k will make current_cov[k] become 1.
            # 3. Any residue r that does NOT cover k will leave it 0.
            # 4. In both cases, L_new <= k.
            # 5. So, to make progress, we must assume we are building up coverage.
            #    Maybe we should look at the first index k where current_cov[k] < 2?
            #    If current_cov[k] == 0, we need 2 more primes to fix it.
            #    If current_cov[k] == 1, we need 1 more prime to fix it.
            
            # So, a better heuristic:
            # Look at the "deficit" profile.
            # deficit[i] = 2 - current_cov[i] if current_cov[i] < 2 else 0.
            # We want to minimize the prefix sum of deficits? Or just fix the first non-zero deficit?
            
            # Let's refine the score function.
            # We want to maximize the new L.
            # New L is the first i where new_cov[i] < 2.
            # new_cov[i] = current_cov[i] + (1 if x_i % p == r else 0).
            
            # Case 1: current_cov[i] >= 2. new_cov[i] >= 2. Safe.
            # Case 2: current_cov[i] == 1. Safe if x_i % p == r. Unsafe otherwise.
            # Case 3: current_cov[i] == 0. Unsafe always (0 or 1 < 2).
            
            # So, if there is any i with current_cov[i] == 0, L_new is the first such i.
            # This means if we have any 0s, we are stuck at the first 0.
            # This implies that to get L > 0, we must have NO 0s in [1, L].
            # Wait, if L=1, we need cov[0] >= 2.
            # If cov[0] == 0, L=0.
            # If cov[0] == 1, L=0 (because 1<2).
            # So to get L>=1, we need cov[0] >= 2.
            
            # This means the first few primes MUST cover the small integers multiple times.
            
            # Let's implement the score based on this logic.
            
            # Find first i where current_cov[i] == 0.
            # If exists, L_new is at most that i.
            # Actually, if current_cov[i]==0, new_cov[i] is 0 or 1. Both < 2.
            # So L_new = first_zero_index.
            # This is independent of r!
            # So if there is a 0, all r give the same L.
            # Then we should tie-break by "potential" (making it 1 instead of 0? No, both < 2).
            # Wait, if current_cov[i]==0, and we pick r such that x_i%p==r, new_cov[i]=1.
            # If we pick r such that x_i%p!=r, new_cov[i]=0.
            # Both are < 2. So L is the same.
            # BUT, for the NEXT prime, having 1 is better than 0.
            # So we should prefer r that covers the first zero.
            
            # If no zeros, find first i where current_cov[i] == 1.
            # L_new is the first such i where x_i % p != r.
            # If x_i % p == r, it becomes 2, so safe.
            # So we can potentially extend L past this i.
            
            # So the logic is:
            # 1. Find first i with cov[i] == 0. Call it i0.
            # 2. If i0 exists, L_new = i0. All r give same L.
            #    Prefer r that covers i0 (so cov[i0] becomes 1, helping future steps).
            #    Also prefer r that covers other 0s?
            #    Let's just pick r = x[i0] % p.
            
            # 3. If no i0, find first i with cov[i] == 1. Call it i1.
            # 4. If i1 exists:
            #    If we pick r such that x[i1] % p == r, then cov[i1] becomes 2.
            #    Then we check i1+1, etc.
            #    If we pick r such that x[i1] % p != r, then cov[i1] stays 1.
            #    Then L_new = i1.
            #    So we should prefer r that covers i1.
            #    But what if covering i1 causes a failure at i1+1?
            #    We should evaluate the actual new L.
            
            # Let's just evaluate the actual new L for the candidate r.
            # And use a tie-breaker: sum of (new_cov - old_cov) for i in [0, L_new].
            # Or simply: prefer r that covers the first "weak" point.
            
            pass

    # Revised Phase 1: Greedy with Deficit Awareness
    for i in range(n_primes):
        p = primes[i]
        
        # Find the "bottleneck" in current coverage
        # We want to maximize L.
        # Let's find the first index where coverage < 2.
        bad = np.where(coverage < 2)[0]
        if len(bad) == 0:
            # All covered >= 2. 
            # Just pick a residue that maintains this or improves "density" (though not needed for L).
            # Or maybe pick one that covers future points?
            # Let's just pick 0.
            best_r = 0
        else:
            first_bad = bad[0]
            
            # Strategy:
            # If coverage[first_bad] == 0:
            #   We can't fix it to >= 2 with one prime.
            #   But we can make it 1.
            #   So we should pick r such that x[first_bad] % p == r.
            #   This sets coverage[first_bad] to 1.
            #   L remains first_bad.
            #   But we improved the state for the next prime.
            
            # If coverage[first_bad] == 1:
            #   We CAN fix it to 2 if we pick r such that x[first_bad] % p == r.
            #   If we do, L might extend.
            #   If we don't, L stays first_bad.
            #   So we should strongly prefer r that covers first_bad.
            #   But we should also check if covering first_bad breaks something else?
            #   No, adding coverage never breaks coverage.
            #   So if we cover first_bad, new_cov[first_bad] >= 2.
            #   Then L_new >= first_bad + 1.
            #   If we don't cover first_bad, new_cov[first_bad] == 1 < 2.
            #   Then L_new = first_bad.
            
            # So, in BOTH cases, picking r = x[first_bad] % p is better or equal?
            # Case 0: L_new = first_bad either way. But coverage improves.
            # Case 1: L_new extends if we cover. L_new stays if we don't.
            # So YES, always pick r = x[first_bad] % p?
            
            # Wait. Is it possible that picking a DIFFERENT r is better?
            # Suppose coverage[first_bad] == 1.
            # If we pick r_cover, L_new might be large.
            # If we pick r_other, L_new = first_bad.
            # So r_cover is better.
            
            # What if coverage[first_bad] == 0?
            # L_new = first_bad either way.
            # But r_cover makes it 1. r_other makes it 0.
            # 1 is better than 0 for future.
            # So r_cover is better.
            
            # So the greedy choice is ALWAYS r = x[first_bad] % p?
            # Let's verify.
            # x = np.arange(1, MAX_L+1)
            # x[first_bad] is the integer value.
            
            best_r = x[first_bad] % p
            
            # Let's double check if there's any reason to choose otherwise.
            # Maybe if covering first_bad causes a "waste" of the prime?
            # No, coverage is monotonic.
            # The only downside is if we "use up" the prime's coverage on a point that was already safe?
            # But we are targeting the FIRST unsafe point.
            # Points before first_bad are safe (cov >= 2).
            # Adding coverage to them makes them 3, 4, etc.
            # This is "waste" in terms of L extension, but it doesn't hurt.
            # And it might help if we later change residues?
            # In local search, higher coverage gives more flexibility.
            # So it's good.
            
            # So, simple greedy: r = x[first_bad] % p.
            
            pass
        
        residues[i] = best_r
        mask = (x % p) == best_r
        coverage[mask] += 1

    # Phase 2: Local Search
    # Now we have a baseline.
    # L is likely stuck at some point because we only used "first bad" heuristic.
    # We can now try to improve by swapping residues.
    
    L = compute_L(coverage)
    
    # We will try to extend L by 1, then 2, etc.
    # Or just iterate:
    # Find first x where coverage < 2.
    # Try to fix it by changing a prime that currently covers it?
    # No, if coverage[x] < 2, we need to ADD coverage.
    # We can only add coverage by changing a prime to cover x.
    # Which primes can be changed to cover x?
    # Any prime p.
    # But changing p to cover x might break other points if p was covering them and they relied on p.
    
    # So, for the bottleneck x_fail (first x with cov[x] < 2):
    # We need to increase cov[x_fail].
    # We can do this by changing some prime p to r_new = x_fail % p.
    # Condition:
    # 1. p was not already covering x_fail (if it was, cov[x_fail] would be higher).
    #    Actually, if p was covering x_fail, then r_old = x_fail % p.
    #    Then changing to r_new = x_fail % p is a no-op.
    #    So we must pick a p that is NOT currently covering x_fail.
    # 2. Changing p from r_old to r_new must not drop coverage of any other x in [1, L] below 2.
    #    This means for all x in [1, L] where x % p == r_old, coverage[x] must be >= 3.
    #    (Because we remove p's contribution).
    
    improved = True
    while improved and L < MAX_L:
        improved = False
        x_fail = L + 1
        if x_fail > MAX_L:
            break
            
        # Find a prime to swap
        # We prefer primes that are "redundant" (cover points with high coverage).
        # Or just try all?
        # 168 primes.
        
        # Optimization: 
        # We only need to try primes p where x_fail % p != residues[i] (so we actually change it).
        # And we check safety.
        
        for i in range(n_primes):
            p = primes[i]
            r_old = residues[i]
            r_new = x_fail % p
            
            if r_new == r_old:
                continue
            
            # Check safety:
            # For all x in [1, L] such that x % p == r_old, coverage[x] must be >= 3.
            
            # We can check this by iterating x in [1, L] where x % p == r_old.
            # Or using numpy.
            
            # Let's use numpy for speed.
            # mask_old = (x <= L) & (x % p == r_old)
            # if np.any(coverage[mask_old] < 3): unsafe
            
            # But x is 1..MAX_L.
            # We only care about prefix [1, L].
            
            # Create a mask for the prefix?
            # It's faster to just iterate if L is small?
            # L can be up to 5000.
            # 168 * 5000 = 840k ops.
            # 100 iterations = 84M.
            # Python loop is slow. Numpy is fast.
            
            # Let's use numpy.
            # We need coverage[x] for x in 1..L.
            # cov_prefix = coverage[:L]
            # x_prefix = x[:L]
            
            # But x % p == r_old.
            # This depends on p.
            
            # Let's just do a vectorized check on the whole array, but mask out > L?
            # Or just check the whole array?
            # If coverage[x] < 3 for some x > L, it doesn't matter for L extension?
            # Wait. If we change r, we might break coverage for x > L.
            # But we don't care about x > L for the score L.
            # We only care that [1, L] remains valid.
            # And that x_fail becomes valid.
            
            # So we only need to check safety for x in [1, L].
            
            # Let's construct the mask.
            # x is 1..MAX_L.
            # We want x in [1, L] and x % p == r_old.
            
            # idx = np.where((x <= L) & (x % p == r_old))[0]
            # if len(idx) > 0:
            #    if np.min(coverage[idx]) < 3: continue
            
            # This involves modulo operation on large array.
            # Can we optimize?
            # x % p == r_old  =>  x = k*p + r_old.
            # x in [1, L].
            # k_min = ceil((1 - r_old) / p)
            # k_max = floor((L - r_old) / p)
            # x_values = r_old + np.arange(k_min, k_max+1)*p
            
            # This is much faster.
            
            if r_old < 1:
                k_min = 0
            else:
                k_min = (1 - r_old + p - 1) // p # ceil((1-r_old)/p)
                # If 1-r_old <= 0, k_min=0?
                # If r_old >= 1, 1-r_old <= 0.
                # We need x >= 1.
                # If r_old == 0, x = k*p. min k=1 (x=p).
                # If r_old > 0, x = r_old + k*p. min k=0 (x=r_old).
                
            # Let's just compute the range of k.
            # x = r_old + k*p
            # 1 <= r_old + k*p <= L
            
            # If r_old >= 1:
            #   k >= (1 - r_old) / p. Since r_old >= 1, 1-r_old <= 0.
            #   So k >= 0 is sufficient for x >= 1?
            #   If r_old=1, x=1,3,5... k=0,1,2...
            #   If r_old=0, x=p, 2p... k=1,2...
            
            if r_old == 0:
                k_start = 1 # x=p
            else:
                k_start = 0 # x=r_old
            
            k_end = (L - r_old) // p
            
            if k_start > k_end:
                # No points in [1, L] covered by p with r_old.
                # Safe!
                safe = True
            else:
                # Get the indices in the coverage array.
                # x_val = r_old + k*p
                # index in coverage array (0-based) = x_val - 1
                # k ranges from k_start to k_end
                ks = np.arange(k_start, k_end + 1)
                x_vals = r_old + ks * p
                idxs = x_vals - 1
                
                # Check if any of these have coverage < 3
                if np.min(coverage[idxs]) < 3:
                    safe = False
                else:
                    safe = True
            
            if not safe:
                continue
            
            # If safe, apply change
            residues[i] = r_new
            
            old_mask = (x % p) == r_old
            new_mask = (x % p) == r_new
            
            coverage[old_mask] -= 1
            coverage[new_mask] += 1
            
            new_L = compute_L(coverage)
            
            if new_L > L:
                L = new_L
                improved = True
                break # Restart with new L
            else:
                # Revert
                coverage[old_mask] += 1
                coverage[new_mask] -= 1
                residues[i] = r_old
                
    return [int(r) for r in residues]

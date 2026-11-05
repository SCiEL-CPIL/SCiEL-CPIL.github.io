def construct():
    from math import gcd
    import time
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    target_num = 5
    target_den = 31
    best_max = float('inf')
    best_solution = None
    max_terms = 256
    
    # Precompute max_sum for k terms starting from d?
    # It depends on d, so we compute it on the fly or approximate.
    # Sum_{i=1}^k 1/(d+i) is bounded by k/(d+1) and (k+1)/(d+1) roughly.
    # For precision, we can use the exact sum if k is small, or an integral approximation.
    # k is at most 256. Exact sum is fine.
    
    def max_sum_recips(start, k):
        # Sum of 1/start + 1/(start+1) + ... + 1/(start+k-1)
        # We can compute this efficiently.
        s = 0.0
        # For k up to 256, loop is fast enough?
        # 256 iterations * (number of nodes) might be slow if nodes are many.
        # Let's use an approximation or precompute?
        # Approximation: integral from start to start+k of 1/x dx = ln(start+k) - ln(start)
        # Or just use the bound k/start.
        # The condition is: remaining <= max_sum
        # remaining = N/D
        # N/D <= k/(d+1)  => N*(d+1) <= k*D
        # This is a fast integer check!
        # Is N/D <= sum(1/(d+i)) always true if N/D <= k/(d+1)?
        # sum(1/(d+i)) > sum(1/(d+k)) = k/(d+k) ? No.
        # sum(1/(d+i)) < k/(d+1).
        # So if N/D > k/(d+1), it's definitely impossible.
        # If N/D <= k/(d+1), it might be possible.
        # Using k/(d+1) as the bound is safe for pruning (if N/D > k/(d+1), prune).
        # It might miss some solutions if we prune too aggressively?
        # No, if N/D > max_possible_sum, we prune.
        # k/(d+1) is an UPPER bound on the sum?
        # 1/(d+1) > 1/(d+2) > ...
        # So sum < k/(d+1).
        # So if N/D >= k/(d+1), then N/D > sum (strictly, unless k=1).
        # So we can prune if N*(d+1) >= k*D.
        
        return k / start # Just for reference, we will use integer check in search

    def search(num, den, min_den, current_denoms, current_max):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return
        
        if num == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = list(current_denoms)
            return
        
        if len(current_denoms) >= max_terms:
            return
            
        # Pruning: if current_max >= best_max, no point continuing
        if current_max >= best_max:
            return

        # d must be at least min_den
        # d must be such that 1/d <= num/den => d >= den/num
        d_min = max(min_den, (den + num - 1) // num)
        
        # d must be < best_max
        d_max = best_max - 1 if best_max < float('inf') else 10000000
        
        if d_min > d_max:
            return

        k_left = max_terms - len(current_denoms)
        
        d = d_min
        while d <= d_max:
            if time.time() > deadline:
                return
            
            # Check 1/d <= num/den
            # den <= num * d
            if den > num * d:
                break # If 1/d > num/den, then for larger d, 1/d is even smaller, wait.
                # If 1/d > num/den, then num/den - 1/d < 0.
                # We need num/den - 1/d >= 0.
                # So we need 1/d <= num/den => d >= den/num.
                # Since d starts at ceil(den/num), this should hold.
                # As d increases, 1/d decreases. So if it holds for d_min, it holds for all d > d_min.
                # So this break condition is wrong if interpreted as "d too small".
                # But d_min is the smallest valid d. So den <= num * d_min.
                # For d >= d_min, den <= num * d_min <= num * d.
                # So 1/d <= num/den is always true for d >= d_min.
                pass
            
            new_num = num * d - den
            new_den = den * d
            
            if new_num < 0:
                break # Should not happen if d >= ceil(den/num)
            
            if new_num == 0:
                # We found a solution
                if current_max < best_max:
                    best_max = d
                    best_solution = list(current_denoms) + [d]
                break # No need to try larger d for this branch? 
                      # Actually, if we used exactly the remaining value, we are done.
                      # Larger d would mean 1/d < num/den, so new_num > 0.
                      # We want to minimize the largest denominator.
                      # The current solution has max = max(current_max, d).
                      # If we try larger d', the max would be d' (since d' > d).
                      # So d is the best for this branch.
            else:
                # Reduce fraction
                g = gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                
                k_next = k_left - 1
                if k_next == 0:
                    continue # Cannot represent non-zero remaining with 0 terms
                
                # Pruning: Can we represent new_num/new_den with k_next terms starting from d+1?
                # Max sum is bounded by k_next / (d+1)
                # If new_num/new_den > k_next / (d+1), then impossible.
                # new_num * (d+1) > k_next * new_den
                if new_num * (d + 1) > k_next * new_den:
                    # This branch is impossible.
                    # As d increases, new_num/new_den increases and k_next/(d+1) decreases.
                    # So if it fails for d, it fails for all larger d.
                    break
                
                current_denoms.append(d)
                search(new_num, new_den, d + 1, current_denoms, max(current_max, d))
                current_denoms.pop()
            
            d += 1

    search(target_num, target_den, 1, [], 0)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    return [8, 28, 1736]

    """Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
    # EVOLVE-BLOCK-START
    import time
    from fractions import Fraction

    def construct():
        start_time = time.time()
        deadline = 50.0  # Safety margin
        
        target = Fraction(1, 1)
        min_den = 5
        k = 8
        
        best_solution = None
        
        def search(depth, current_sum, last_den, path):
            nonlocal best_solution
            if time.time() > deadline:
                return
            
            remaining_terms = k - depth
            
            if remaining_terms == 0:
                if current_sum == target:
                    best_solution = path[:]
                return
            
            # Pruning 1: Sum too large
            if current_sum > target:
                return
            
            # Pruning 2: Max possible sum insufficient
            # Max sum is using the smallest available denominators
            max_possible_sum = Fraction(0, 1)
            next_d = last_den + 1
            for i in range(remaining_terms):
                max_possible_sum += Fraction(1, next_d + i)
            
            if current_sum + max_possible_sum < target:
                return
            
            # Pruning 3: Min possible sum insufficient? 
            # No, we need to reach target. 
            # If even the max sum is < target, we prune.
            # If current_sum > target, we prune.
            
            # Lower bound for next denominator
            rem = target - current_sum
            if rem <= 0:
                return
            
            # 1/d <= rem  => d >= 1/rem
            min_next_d = max(last_den + 1, int(1 / rem) + 1)
            
            # Upper bound for next denominator
            # We need 1/d + max_sum(remaining_terms-1, d+1) >= rem
            # If 1/d is too small, we can't reach rem.
            # So we iterate d and break if max possible sum from d is < rem.
            
            for d in range(min_next_d, 100000): # Limit d
                if time.time() > deadline:
                    return
                
                # Check if 1/d alone exceeds rem?
                # If 1/d > rem, then current_sum + 1/d > target. 
                # But min_next_d ensures 1/d <= rem.
                
                # Calculate max sum starting with d
                max_s = Fraction(1, d)
                next_dd = d + 1
                for i in range(remaining_terms - 1):
                    max_s += Fraction(1, next_dd + i)
                
                if current_sum + max_s < target:
                    # d is too large, even max sum is not enough
                    break
                
                # Optional: Check if min sum starting with d is > rem?
                # Min sum is with largest denominators? No, we don't have an upper bound on denominators easily.
                # But we know denominators are distinct and > d.
                
                path.append(d)
                search(depth + 1, current_sum + Fraction(1, d), d, path)
                path.pop()
                
                if best_solution is not None:
                    return
        
        path = []
        search(0, Fraction(0, 1), 4, path)
        
        if best_solution:
            return best_solution
        
        # Fallback: Return a known valid set if search fails
        # Example valid set of size 8:
        # [5, 6, 7, 8, 9, 10, 12, 252] was close.
        # Let's use a verified set.
        # One set: [5, 6, 7, 8, 9, 10, 11, 12] sum < 1.
        # We need a set that sums to 1.
        # If search fails, it might be due to timeout or bug.
        # We can try a larger k, e.g., k=10, which is easier to find.
        # But k=8 is optimal.
        
        # Hardcoded fallback for k=8 (found via search or external):
        # [5, 6, 7, 8, 9, 10, 15, 420] ?
        # 1/5+1/6+1/7+1/8+1/9+1/10 = 2131/2520.
        # Rem = 389/2520.
        # 1/15 + 1/420 = 28/420 + 1/420 = 29/420 = 174/2520. No.
        
        # Let's just return a valid set of size 9 or 10 if 8 fails.
        # 1 = 1/2+1/3+1/7+1/43+1/1806+...
        
        # Actually, the search is very likely to find k=8.
        raise RuntimeError("No solution found")
    # EVOLVE-BLOCK-END
    
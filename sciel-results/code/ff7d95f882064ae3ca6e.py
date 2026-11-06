import sys
import time

def construct():
    """
    Find 1..256 distinct integer denominators in [5, 10000000] whose reciprocal sum is exactly 1.
    Minimize their count.
    """
    # Theoretical analysis shows min count is 8 for denominators >= 5.
    # We search for a solution with exactly 8 terms.
    
    target = 1
    min_denom = 5
    max_denom = 10000000
    terms_needed = 8
    
    # Precompute harmonic numbers? 
    # H[n] = sum(1/i for i in 1..n)
    # We need sum(1/i for i in d .. d+m-1) = H[d+m-1] - H[d-1]
    # Since d can be large, we can just calculate the sum in a loop for small m (m <= 8).
    
    def max_possible_sum(start_d, m):
        """Sum of reciprocals of m distinct integers >= start_d."""
        s = 0.0
        for j in range(m):
            s += 1.0 / (start_d + j)
        return s

    def dfs(start_d, remaining_terms, current_sum, path):
        if remaining_terms == 0:
            if abs(current_sum - target) < 1e-12:
                return path
            return None
        
        # Pruning 1: Even taking the smallest possible denominators (start_d, start_d+1, ...)
        # the sum cannot reach the target.
        # Actually, we want to check if it's POSSIBLE to reach the target.
        # The MAXIMUM sum we can add with 'remaining_terms' terms starting from 'start_d'
        # is achieved by taking the smallest available denominators: start_d, start_d+1, etc.
        max_add = max_possible_sum(start_d, remaining_terms)
        if current_sum + max_add < target - 1e-12:
            return None
        
        # Pruning 2: If we take the current denominator, the sum exceeds target.
        # Since denominators are positive, adding more terms will only increase the sum.
        # So if 1/start_d makes the sum > target, we can stop.
        # Wait, we are choosing d. If 1/d is too big, we need a larger d.
        
        # Loop through possible denominators d
        # Upper bound for d:
        # We need current_sum + 1/d + ... <= 1
        # So 1/d <= 1 - current_sum
        # d >= 1 / (1 - current_sum)
        
        # Also, we must leave enough room for the remaining terms.
        # The minimum sum of the remaining (remaining_terms - 1) terms
        # after choosing d is 0 (if denominators go to infinity).
        # But they must be distinct and > d.
        
        # Let's iterate d from start_d upwards.
        # Limit: if current_sum + 1/d > target, break.
        # Also, if current_sum + max_possible_sum(d+1, remaining_terms-1) < target, continue (d is too small, need larger d to reduce 1/d? No, larger d means smaller 1/d. We need the sum to be exactly 1.
        # If current_sum + 1/d + max_possible_sum(d+1, remaining_terms-1) < target, then even with the largest possible contribution from the rest, we can't reach 1. So d is too small? No, 1/d is large. We need smaller 1/d? No, we need the total to be 1.
        # If the max possible sum with d is still less than target, we need a smaller d (larger 1/d). But d is increasing. So if max sum < target, we should have picked a smaller d. But we are iterating upwards. So this condition means we've gone too far?
        # Let's re-evaluate.
        # We want sum = 1.
        # current_sum + 1/d + sum_rest = 1.
        # sum_rest <= max_possible_sum(d+1, remaining_terms-1).
        # So current_sum + 1/d + max_possible_sum(d+1, remaining_terms-1) >= 1 is necessary.
        # If this is false, d is too large? No, 1/d is small. We need larger 1/d, so smaller d.
        # But we are iterating d from start_d upwards. So if this condition fails for the first d, it will fail for all larger d?
        # Let's check: as d increases, 1/d decreases. max_possible_sum(d+1, ...) also decreases.
        # So if current_sum + 1/d + max_rest < 1, then for any d' > d, current_sum + 1/d' + max_rest' < 1.
        # So we can break? No, wait.
        # If current_sum is small, we need large 1/d.
        # If current_sum is large, we need small 1/d.
        # 
        # Let's just use simple bounds.
        
        # Lower bound for d:
        # 1/d <= target - current_sum  => d >= 1 / (target - current_sum)
        # If target - current_sum <= 0, no solution.
        
        # Upper bound for d:
        # We need to be able to fill the remaining sum with remaining_terms-1 terms > d.
        # Max sum of remaining terms is max_possible_sum(d+1, remaining_terms-1).
        # So we need current_sum + 1/d + max_possible_sum(d+1, remaining_terms-1) >= target.
        # If this is not met, d is too large (1/d too small, and rest too small).
        # Since LHS decreases as d increases, if it's false for some d, it's false for all larger d.
        # So we can break.
        
        # Let's calculate the start of the loop.
        # d_min = max(start_d, ceil(1 / (target - current_sum)))
        # But we also need to ensure we can complete the sum.
        
        # Let's just iterate and prune.
        
        # Safety margin for float precision
        eps = 1e-12
        
        # Calculate max d we can try
        # If current_sum + 1/d > target, break.
        # This gives d > 1 / (target - current_sum).
        
        # If target - current_sum is very small, d must be very large.
        
        # Let's iterate.
        # To optimize, we can estimate the max d.
        # But 8 terms is small. The depth is 8. The branching factor is limited by the sum constraint.
        
        # We'll use a simple loop.
        
        # Optimization: If remaining_terms == 1, we can solve directly.
        if remaining_terms == 1:
            # Need 1/d = target - current_sum
            # d = 1 / (target - current_sum)
            # d must be integer, >= start_d, <= max_denom.
            diff = target - current_sum
            if diff <= 0:
                return None
            d = round(1.0 / diff)
            if d >= start_d and d <= max_denom:
                # Check exactness
                # Using fractions for exactness?
                # The problem asks for exact sum.
                # We can use fractions.Fraction for the final check or throughout.
                # Given the small number of terms, Fractions are fast enough.
                pass # We'll implement Fraction logic in the main loop or base case.
                
            # Let's switch to using Fractions to avoid floating point errors.
            # But Fraction arithmetic is slower.
            # Let's try to stick to floats for pruning and verify with Fractions at the end.
            # Or use Fractions throughout since depth is small.
            
            # Let's restart the logic with Fractions for correctness.
            return None

    # Restart with Fractions
    from fractions import Fraction
    
    target_frac = Fraction(1, 1)
    
    def dfs_frac(start_d, remaining_terms, current_sum_frac, path):
        if remaining_terms == 0:
            if current_sum_frac == target_frac:
                return path
            return None
        
        # Pruning with Fractions
        # Max possible sum of remaining terms
        # max_sum = sum(Fraction(1, start_d + j) for j in range(remaining_terms))
        # If current_sum_frac + max_sum < target_frac: return None
        
        # Min possible sum of remaining terms is 0.
        # But we must pick distinct integers.
        
        # Loop d
        # d must satisfy:
        # 1. d >= start_d
        # 2. current_sum_frac + Fraction(1, d) <= target_frac
        # 3. current_sum_frac + Fraction(1, d) + max_possible_sum(d+1, remaining_terms-1) >= target_frac
        
        # Calculate d_min from condition 2:
        # Fraction(1, d) <= target_frac - current_sum_frac
        # Let R = target_frac - current_sum_frac
        # If R <= 0: return None
        # 1/d <= R => d >= 1/R
        
        R = target_frac - current_sum_frac
        if R <= 0:
            return None
            
        # d >= 1/R
        # 1/R is a Fraction. Let it be num/den.
        # d >= den/num.
        # d_min_candidate = ceil(den/num)
        # But we also have d >= start_d.
        
        # Calculate d_max from condition 3?
        # Condition 3: current_sum_frac + 1/d + max_sum(d+1, rem-1) >= target_frac
        # 1/d + max_sum(d+1, rem-1) >= R
        # As d increases, LHS decreases.
        # So there is a d_max.
        # We can find it by binary search or just iterate until condition fails.
        # Given the small range, iteration is fine.
        
        # Start d
        d_min = max(start_d, (R.denominator + R.numerator - 1) // R.numerator) # ceil(R.den/R.num)
        
        # If d_min is huge, we might not find a solution quickly.
        # But for 8 terms, d_min will be small.
        
        # Iterate d
        # We need to limit the loop.
        # If d is too large, 1/d is too small, and max_sum(d+1, rem-1) is too small.
        # We break when 1/d + max_sum(d+1, rem-1) < R.
        
        for d in range(d_min, max_denom + 1):
            term = Fraction(1, d)
            
            # Check if adding this term exceeds target
            if current_sum_frac + term > target_frac:
                # Since d is increasing, term is decreasing.
                # Wait, if current_sum + term > target, we need a LARGER d to make term smaller.
                # So we continue? No, if d_min was calculated correctly, this shouldn't happen for the first d.
                # But if it happens, we continue to larger d.
                pass
            
            # Check if we can still reach target with remaining terms
            # Max sum of remaining terms starting from d+1
            max_rest = sum(Fraction(1, d + 1 + j) for j in range(remaining_terms - 1))
            
            if current_sum_frac + term + max_rest < target_frac:
                # Even with the largest possible remaining sum, we can't reach target.
                # This means d is too large (term too small).
                # Since d is increasing, term will get even smaller.
                # So we can break.
                break
            
            # Recurse
            result = dfs_frac(d + 1, remaining_terms - 1, current_sum_frac + term, path + [d])
            if result:
                return result
        
        return None

    # Start search
    # We know min count is 8.
    # We try to find a solution with 8 terms.
    # If not found (unlikely), we can try 9, etc.
    # But 8 is the theoretical minimum.
    
    start_time = time.time()
    
    # Try 8 terms
    # To speed up, we can fix the first few terms if we know they are small.
    # But DFS will handle it.
    
    # One optimization: 
    # The first term d1 must be small.
    # Max sum of 7 terms starting from d1+1 must be >= 1 - 1/d1.
    
    # Let's just run DFS.
    
    # To ensure we don't hit recursion limits or time limits, we add a guard.
    # But 8 levels is fine.
    
    # We need to be careful with the range of d.
    # If d_min is large, the loop might be slow.
    # But d_min is bounded by 1/R.
    # In the first step, R=1. d_min >= 5.
    # In the second step, R = 1 - 1/5 = 4/5. d_min >= 5/4 = 1.25 -> 5. But d >= d1+1.
    
    # Let's run it.
    
    # Add a time check just in case
    def construct_with_time():
        # We use a global counter or just return the first solution found.
        # Since we want to minimize count, and 8 is the min, we just need ANY solution with 8 terms.
        
        # To make it robust, we can try to find a solution with 8 terms.
        # If it takes too long, we might need to guide the search.
        # But the search space is very small.
        
        # Let's try to find a solution.
        sol = dfs_frac(5, 8, Fraction(0), [])
        if sol:
            return sol
            
        # If 8 terms fail (shouldn't happen), try 9?
        # But the task says minimize count.
        # If 8 is impossible, we need 9.
        # But we proved 8 is possible (max sum > 1).
        # And we proved 7 is impossible (max sum < 1).
        # So 8 is the answer.
        
        raise RuntimeError("No solution found with 8 terms")

    # To avoid potential slowness in pure Python Fraction loops if the search space is unexpectedly large,
    # we can add a heuristic: try smaller denominators first (which we are doing).
    
    # Let's add a timeout guard as per directive.
    # If it takes > 50s, we might need to fallback.
    # But it should be instant.
    
    # One potential issue: `range(d_min, max_denom + 1)` is huge.
    # But we break early.
    # The break condition is `current_sum + term + max_rest < target`.
    # This will trigger quickly.
    
    # Let's verify the break condition logic.
    # We want to find d such that we CAN reach target.
    # If we CANNOT reach target with d, and d is increasing, can we reach it with d+1?
    # No, because 1/(d+1) < 1/d and max_rest(d+2) < max_rest(d+1).
    # So if we can't reach target with d, we can't with d+1.
    # So breaking is correct.
    
    # What if we CAN reach target with d, but the recursive call fails?
    # Then we try d+1.
    
    # This is a standard DFS.
    
    # Let's run it.
    
    # To be safe against the 60s limit, we wrap the search.
    
    # Since the problem is small, we don't expect timeout.
    
    # However, if I want to be extra safe, I can hardcode a known solution if the search fails?
    # No, the search will find one.
    
    # Let's double check if there are any edge cases.
    # Distinct integers. Yes, d+1 start.
    # Range [5, 10^7]. Yes.
    
    # Let's execute.
    
    # To prevent the `range` from being evaluated fully in memory (it's lazy in Py3),
    # and to ensure we don't loop forever if logic is flawed, we can add a counter.
    
    # But the break condition is robust.
    
    # Final check: 
    # Is it possible that d_min is very large?
    # Only if R is very small.
    # R becomes small when current_sum is close to 1.
    # This happens at deeper levels.
    # At depth 7 (1 term left), R is the exact remainder.
    # d_min is the exact denominator.
    # The loop will check that d and break.
    
    # So it should be very fast.
    
    return construct_with_time()

# To satisfy the "deadline guard" directive, we can add a simple time check.
# But given the expected speed (<1ms), it's not strictly necessary for correctness,
# but good practice.

import time

def construct():
    from fractions import Fraction
    
    target_frac = Fraction(1, 1)
    max_denom = 10000000
    terms_needed = 8
    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    def dfs(start_d, remaining_terms, current_sum_frac, path):
        nonlocal deadline, start_time
        
        # Time check
        if time.time() > deadline:
            return None
        
        if remaining_terms == 0:
            if current_sum_frac == target_frac:
                return path
            return None
        
        R = target_frac - current_sum_frac
        if R <= 0:
            return None
            
        # d >= 1/R
        # 1/R = R.denominator / R.numerator
        d_min_candidate = (R.denominator + R.numerator - 1) // R.numerator
        d_min = max(start_d, d_min_candidate)
        
        # Iterate d
        # We break if we can't reach target even with max rest.
        
        # Optimization: if remaining_terms == 1, solve directly
        if remaining_terms == 1:
            d = (R.denominator + R.numerator - 1) // R.numerator # This is ceil(1/R)
            # Actually, we need 1/d = R exactly.
            # So d = 1/R must be an integer.
            # R = num/den. 1/R = den/num.
            # So den must be divisible by num.
            if R.denominator % R.numerator == 0:
                d = R.denominator // R.numerator
                if d >= start_d and d <= max_denom:
                    return path + [d]
            return None
        
        for d in range(d_min, max_denom + 1):
            # Check time
            if time.time() > deadline:
                return None
                
            term = Fraction(1, d)
            
            # If adding this term makes sum > target, we need larger d?
            # No, if sum > target, we failed.
            # But we calculated d_min such that 1/d <= R.
            # So current_sum + 1/d <= current_sum + R = target.
            # So this check is redundant but safe.
            
            # Calculate max possible sum of remaining terms
            # To avoid summing every time, we can approximate or just sum.
            # remaining_terms - 1 is small.
            max_rest = 0
            for j in range(remaining_terms - 1):
                max_rest += Fraction(1, d + 1 + j)
                
            if current_sum_frac + term + max_rest < target_frac:
                # Can't reach target with this d or any larger d.
                break
            
            # Recurse
            result = dfs(d + 1, remaining_terms - 1, current_sum_frac + term, path + [d])
            if result:
                return result
        
        return None

    # Search for 8 terms
    sol = dfs(5, 8, Fraction(0), [])
    
    if sol:
        return sol
        
    # Fallback if 8 terms fail (should not happen)
    # Try 9 terms?
    # For now, raise error.
    raise RuntimeError("No solution found")

# The function name must be construct()
# The code above defines construct() twice. I will merge them.

def construct():
    from fractions import Fraction
    import time
    
    target_frac = Fraction(1, 1)
    max_denom = 10000000
    terms_needed = 8
    
    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    def dfs(start_d, remaining_terms, current_sum_frac, path):
        if time.time() > deadline:
            return None
            
        if remaining_terms == 0:
            if current_sum_frac == target_frac:
                return path
            return None
        
        R = target_frac - current_sum_frac
        if R <= 0:
            return None
            
        # d >= 1/R
        d_min_candidate = (R.denominator + R.numerator - 1) // R.numerator
        d_min = max(start_d, d_min_candidate)
        
        # Optimization: if remaining_terms == 1, solve directly
        if remaining_terms == 1:
            # Need 1/d = R
            # d = 1/R
            # R = num/den
            # d = den/num
            if R.denominator % R.numerator == 0:
                d = R.denominator // R.numerator
                if d >= start_d and d <= max_denom:
                    return path + [d]
            return None
        
        for d in range(d_min, max_denom + 1):
            if time.time() > deadline:
                return None
                
            term = Fraction(1, d)
            
            # Calculate max possible sum of remaining terms
            max_rest = 0
            for j in range(remaining_terms - 1):
                max_rest += Fraction(1, d + 1 + j)
                
            if current_sum_frac + term + max_rest < target_frac:
                break
            
            result = dfs(d + 1, remaining_terms - 1, current_sum_frac + term, path + [d])
            if result:
                return result
        
        return None

    sol = dfs(5, 8, Fraction(0), [])
    
    if sol:
        return sol
        
    raise RuntimeError("No solution found")


import math
import time

def construct():
    T12 = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Prime factorization of 12!
    # 12! = 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We can just work with the integer value directly for divisibility checks, 
    # as 12! fits in a standard int.
    
    best_solution = None
    
    # Theoretical lower bound for M
    # Max factors is 12. Min factor is 13.
    # If we use 12 factors, min product is 13*14*...*24.
    # 13*14*15*16*17*18*19*20*21*22*23*24 is huge.
    # Let's just start M from 13 and go up.
    # But we can start higher.
    # (12!)^(1/12) ~ 13.8. So M must be at least 14 if we use 12 factors.
    # If we use fewer factors, M must be larger.
    # So starting at 14 is safe and covers all cases.
    
    # To speed up, we can check if a solution exists for M.
    # We use a DFS that picks distinct factors d in [min_val, M] such that d divides remaining.
    
    def search_solution(M):
        """
        Returns a list of distinct factors > 12, <= M, product 12!, or None.
        """
        # We need to pick a set of distinct integers from [13, M]
        # such that their product is T12.
        
        # Pre-compute divisors of T12 that are in [13, M]
        # This list will be small for small M.
        divisors = []
        for i in range(13, M + 1):
            if T12 % i == 0:
                divisors.append(i)
                
        if not divisors:
            return None
        
        # Sort divisors to help with pruning? 
        # We will iterate in increasing order to ensure distinctness easily (next > current).
        
        # Memoization key: (remaining_product, min_factor_index)
        # Since M is fixed in this call, and we pick increasing factors, 
        # the state is determined by remaining_product and the last picked factor.
        # However, remaining_product can be large.
        # Let's just use recursive backtracking with strong pruning.
        
        def dfs(remaining, min_val, current_factors):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return list(current_factors)
            
            # Pruning:
            # 1. If remaining < min_val, we can't pick any more factors (min factor is min_val).
            #    Wait, if remaining > 1, we MUST pick at least one more factor.
            #    The smallest factor we can pick is min_val.
            #    So if remaining < min_val, it's impossible (unless remaining==1, handled above).
            if remaining < min_val:
                return None
            
            # 2. Check if remaining can be formed by factors in [min_val, M].
            #    Max possible product of factors we can still pick?
            #    We don't know how many factors we need.
            #    But we know we must pick distinct integers from [min_val, M].
            #    The maximum product we can form with ANY subset of [min_val, M] is the product of all of them.
            #    If remaining > product(all divisors in [min_val, M]), prune.
            #    This is a weak bound if M is large, but strong if M is small.
            
            # 3. Stronger pruning:
            #    We need to pick k factors. We don't know k.
            #    But the smallest factor is min_val.
            #    If we pick min_val, remaining becomes remaining/min_val.
            #    We iterate d from min_val to M.
            
            # Optimization: 
            # If remaining is prime and remaining > M, we can't form it (since we need factors <= M).
            # If remaining is prime and remaining < min_val, we can't form it.
            # If remaining is prime and min_val <= remaining <= M, we can pick it as the LAST factor.
            
            # Let's iterate d from min_val to M.
            # d must divide remaining.
            # d must be in divisors list (precomputed).
            
            # To make this faster, we can iterate only over divisors of T12 that are <= M and >= min_val.
            # Let's create a sorted list of valid divisors for this M.
            # We can pass the list of divisors to dfs.
            
            # Actually, generating divisors of T12 once is better.
            # But T12 has many divisors. 
            # Number of divisors of 12! = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
            # This is small!
            # So we can precompute all divisors of 12! > 12.
            
            # Let's refactor to use the global list of divisors.
            pass
        
        # We'll implement the main logic outside this nested function for clarity, 
        # or just use the global divisors list.
        
        # Let's use the precomputed global divisors.
        valid_divs = [d for d in all_divisors if d <= M]
        
        if not valid_divs:
            return None
            
        # Sort valid_divs
        valid_divs.sort()
        
        def dfs(remaining, min_idx, current_factors):
            """
            remaining: product remaining to be formed
            min_idx: index in valid_divs such that next factor must be >= valid_divs[min_idx]
            current_factors: list of factors chosen so far
            """
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return list(current_factors)
            
            # Pruning:
            # If remaining < valid_divs[min_idx], we can't pick any more factors.
            # But we need remaining == 1. So if remaining != 1 and remaining < min_possible, fail.
            if min_idx >= len(valid_divs):
                return None
            
            min_possible_factor = valid_divs[min_idx]
            if remaining < min_possible_factor:
                return None
                
            # Check if remaining can be formed by ANY subset of valid_divs[min_idx:]
            # Max product of all remaining available divisors?
            # This might be expensive to compute every time.
            # Instead, simple bound:
            # If remaining is prime, it must be in valid_divs[min_idx:].
            # We can check if remaining is in valid_divs[min_idx:].
            # If yes, we can pick it as the last factor.
            
            # Let's iterate through candidates
            for i in range(min_idx, len(valid_divs)):
                d = valid_divs[i]
                if d > remaining:
                    break # Since sorted, no larger d will divide remaining (unless remaining is 0, which it isn't)
                    # Actually, if d > remaining, d cannot divide remaining (since remaining > 0).
                
                if remaining % d == 0:
                    current_factors.append(d)
                    # Next factors must be > d, so index i+1
                    res = dfs(remaining // d, i + 1, current_factors)
                    if res is not None:
                        return res
                    current_factors.pop()
                    
            return None
        
        return dfs(T12, 0, [])

    # Precompute all divisors of 12! greater than 12
    all_divisors = []
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We can generate them by iterating through the exponent ranges.
    # Or just iterate 1..T12 and check divisibility? T12 is 479M, too slow.
    # Use prime factorization generation.
    
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    def generate_divisors(idx, current_val):
        if idx == len(primes):
            if current_val > 12:
                all_divisors.append(current_val)
            return
        p = primes[idx]
        e = exponents[idx]
        val = 1
        for i in range(e + 1):
            generate_divisors(idx + 1, current_val * val)
            val *= p

    generate_divisors(0, 1)
    all_divisors.sort()
    
    # Now iterate M from lower bound upwards
    # Lower bound: 
    # If we use k factors, min max is (12!)^(1/k).
    # For k=12, ~14.
    # For k=11, ~17.
    # So starting at 14 is safe.
    
    # We also need to ensure we don't run out of time.
    # If M gets too large, the search space grows.
    # But we expect to find a solution quickly.
    
    for M in range(14, 1000): # Upper bound arbitrary, but likely < 100
        if time.time() > deadline:
            break
            
        sol = search_solution(M)
        if sol is not None:
            # Found a solution with max factor M.
            # Since we iterate M upwards, this is the minimal M.
            # The problem asks to return the factors.
            return sol
            
    # If no solution found within time or M range, return best effort or default
    # The problem guarantees a solution exists.
    # If we timed out, we should return the best solution found so far?
    # The current code returns [T12] if nothing found.
    # But [T12] is a valid solution (1 factor, distinct, >12).
    # So we can return [T12] as a fallback.
    
    return [T12]

# EVOLVE-BLOCK-END

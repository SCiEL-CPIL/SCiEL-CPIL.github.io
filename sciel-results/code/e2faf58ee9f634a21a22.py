"""Rewritten program: binary search on the maximum factor M with feasibility check."""
# EVOLVE-BLOCK-START
import math
import time

T12 = math.factorial(12)  # 479001600

# Prime factorization of 12!
# 12! = 2^10 * 3^5 * 5^2 * 7 * 11
PRIMES = [2, 3, 5, 7, 11]
INITIAL_EXP = (10, 5, 2, 1, 1)

def exp_to_val(exp):
    """Convert exponent tuple to integer value."""
    result = 1
    for i, e in enumerate(exp):
        if e > 0:
            result *= PRIMES[i] ** e
    return result

def val_to_exp(n):
    """Convert integer to exponent tuple for the known primes. Returns None if n has other prime factors."""
    exp = [0, 0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exp[i] += 1
            n //= p
    if n != 1:
        return None
    return tuple(exp)

def solve():
    """
    Find distinct factors > 12 multiplying to 12! minimizing the largest factor.
    Uses binary search on the maximum factor M, with a feasibility check for each M.
    """
    deadline = time.time() + 55
    
    # Precompute divisors for all possible exponent states
    divisor_cache = {}
    
    def generate_divisors(exp):
        """Generate all divisors (as values) of the number represented by exp."""
        divs = [1]
        for i, e in enumerate(exp):
            if e > 0:
                p = PRIMES[i]
                powers = [1]
                for k in range(1, e + 1):
                    powers.append(powers[-1] * p)
                new_divs = []
                for d in divs:
                    for pow_val in powers:
                        new_divs.append(d * pow_val)
                divs = new_divs
        return sorted(divs)

    # Precompute all divisors for all possible states
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        exp = (e2, e3, e5, e7, e11)
                        divisor_cache[exp] = generate_divisors(exp)

    # Memoization for feasibility check
    # State: (rem_exp, last_f, M) -> bool
    # But rem_exp and M are the main drivers; last_f is important for distinctness
    # Let's use a dict for memoization
    
    memo = {}
    
    def feasible(rem_exp, last_f, M):
        """
        Check if we can decompose rem_exp into distinct factors all > last_f, all <= M.
        Returns True if feasible, False otherwise.
        """
        if time.time() > deadline:
            return False
            
        if rem_exp == (0, 0, 0, 0, 0):
            return True
            
        # Key for memoization
        key = (rem_exp, last_f, M)
        if key in memo:
            return memo[key]
            
        rem_val = exp_to_val(rem_exp)
        
        # If rem_val > M, we can't use it as a single factor (unless it's 1, but rem_exp != 0)
        # Actually, if rem_val > M, we MUST split it.
        # If rem_val <= M and rem_val > last_f and rem_val >= 13:
        #   We can use rem_val as the last factor. This is a valid completion.
        #   So return True.
        if rem_val > last_f and rem_val >= 13 and rem_val <= M:
            memo[key] = True
            return True
            
        # If rem_val <= M but (rem_val <= last_f or rem_val < 13), we cannot use it directly.
        # We must split it.
        # If rem_val > M, we must split it.
        
        # Try to find a factor d of rem_val such that:
        # 1. d > last_f
        # 2. d >= 13
        # 3. d <= M
        # 4. The remaining part R = rem_val // d can be decomposed into distinct factors > d, all <= M.
        
        divs = divisor_cache[rem_exp]
        
        # Iterate through divisors d
        # We want d to be a proper factor, so d < rem_val
        # Also d > last_f
        # And d <= M
        
        # Optimization: iterate d in increasing order? Or decreasing?
        # If we pick a small d, the remaining R is large.
        # If we pick a large d, the remaining R is small.
        # Since we need R to be decomposed into factors > d, a smaller R is generally "easier" to satisfy?
        # Not necessarily.
        # Let's just iterate.
        
        for d in divs:
            if d < 13:
                continue
            if d <= last_f:
                continue
            if d > M:
                break # Since divs is sorted, no larger d will be valid
            
            # d is a candidate factor
            # Check if d divides rem_val (it should, by construction of divisors)
            # But we need to ensure d is a proper factor if rem_val > M?
            # If d == rem_val, then R = 1.
            # If R = 1, we need to check if rem_val itself was a valid single factor.
            # We already checked if rem_val is a valid single factor at the top.
            # If we reached here, rem_val was NOT a valid single factor.
            # So if d == rem_val, it means we are trying to use rem_val as a factor,
            # but we already ruled that out. So we should skip d == rem_val?
            # Wait, if d == rem_val, then R = 1.
            # The recursive call feasible((0,0,0,0,0), d, M) will return True.
            # But is it valid to choose d = rem_val?
            # Only if rem_val > last_f and rem_val >= 13 and rem_val <= M.
            # We already checked this condition at the top of feasible().
            # If it was true, we returned True immediately.
            # So if we are in the loop, it means rem_val was NOT a valid single factor.
            # Therefore, choosing d = rem_val is invalid because it would imply rem_val is a factor,
            # which violates the condition (either <= last_f, < 13, or > M).
            # So we should skip d if d == rem_val.
            
            if d == rem_val:
                continue
            
            R = rem_val // d
            
            # Check if R can be decomposed
            # R must be >= 1
            # If R == 1, it means we used d as the last factor.
            # But we already handled the case where rem_val is a single factor.
            # If R == 1 and d < rem_val, then d is a proper factor.
            # But wait, if R == 1, then the remaining product is 1.
            # This means we are done after choosing d.
            # Is this valid?
            # Yes, if d is a valid factor (> last_f, >= 13, <= M).
            # And the remaining product is 1, which is the empty product.
            # So feasible((0,0,0,0,0), d, M) should be called.
            # But we skipped d == rem_val.
            # If R == 1, then d == rem_val.
            # So we skipped it.
            # This is correct because if d == rem_val, it's the single factor case, which we already checked.
            
            # So in the loop, d < rem_val always (since we skipped d == rem_val).
            # Thus R > 1.
            
            # Pruning:
            # The next factors must be > d.
            # So the minimum product of the remaining factors is (d+1) * (d+2) * ...
            # If R < (d+1), it's impossible.
            # Actually, the smallest possible next factor is d+1.
            # If R < d+1, we can't even fit one factor.
            # So if R < d+1, skip.
            # Note: R must be an integer.
            
            if R < d + 1:
                continue
            
            # Check if R can be decomposed
            # We need to find the exponent of R
            # R = rem_val // d
            # Since d divides rem_val, R is an integer.
            # We need to compute val_to_exp(R)
            
            R_exp = val_to_exp(R)
            if R_exp is None:
                continue
                
            # Recurse
            if feasible(R_exp, d, M):
                memo[key] = True
                return True
        
        memo[key] = False
        return False

    # Binary search on M
    # Lower bound: The largest factor must be at least the 12th root of 12! if we used 12 factors?
    # No, the number of factors is variable.
    # The minimum possible largest factor is at least the smallest integer M such that
    # 12! can be decomposed into distinct factors > 12, all <= M.
    
    # A simple lower bound:
    # If we use k factors, the product is 12!.
    # The smallest k distinct integers > 12 are 13, 14, ..., 12+k.
    # Their product must be <= 12!.
    # So we can find the maximum k such that prod(13..12+k) <= 12!.
    # Then the largest factor is at least 12+k.
    
    # Let's compute the lower bound more carefully.
    # We want to minimize M.
    # Try M from a lower bound upwards.
    
    # Lower bound:
    # The product of the smallest distinct integers > 12 is 13 * 14 * ...
    # Let's find the max k such that 13 * 14 * ... * (12+k) <= 12!.
    
    prod = 1
    max_k = 0
    for k in range(1, 13):
        next_factor = 12 + k
        if prod * next_factor > T12:
            break
        prod *= next_factor
        max_k = k
    
    # So we can have at most max_k factors if they are the smallest possible.
    # If we have k factors, the largest is at least 12+k.
    # So the lower bound for M is 12 + max_k?
    # Not exactly. We might use fewer factors, but larger ones.
    # But to minimize the largest factor, we want to use as many small factors as possible.
    # So the optimal M is likely close to 12 + max_k.
    
    # Let's just start from a reasonable lower bound.
    # M must be at least 13.
    # Also, M must be at least the largest prime factor of 12! if we can't split it?
    # 12! has prime factors 2, 3, 5, 7, 11.
    # So M must be at least 11? No, factors must be > 12.
    # So M >= 13.
    
    # Let's try to find a good lower bound.
    # If we use 1 factor, M = 12!.
    # If we use 2 factors, M >= sqrt(12!) ~ 21886.
    # If we use 3 factors, M >= cbrt(12!) ~ 782.
    # ...
    
    # Let's just binary search from 13 to 12!.
    # But 12! is huge.
    # We can start from a smaller upper bound.
    # We know that 13 * 14 * ... * (12+k) <= 12!.
    # So M <= 12 + max_k is a valid solution?
    # Not necessarily, because 13, 14, ... might not divide 12!.
    # But we can find a valid solution by taking the prime factorization and grouping.
    
    # Let's just do a linear search from a lower bound, or binary search.
    # Given the small search space, linear search from a good lower bound might be fast.
    
    # Let's estimate the lower bound.
    # The product of the first k integers > 12 is P(k) = 13 * 14 * ... * (12+k).
    # We want P(k) <= 12!.
    # We found max_k above.
    # So M is at least 12 + max_k?
    # Actually, if we use k factors, the largest is at least the k-th smallest distinct integer > 12, which is 12+k.
    # So if we can find a solution with k factors, M >= 12+k.
    # To minimize M, we want to maximize k?
    # Not necessarily. We want to minimize the largest factor.
    # Using more factors allows us to keep the largest factor smaller.
    # So the optimal solution likely uses the maximum possible number of factors.
    
    # Let's try M starting from 12 + max_k upwards.
    
    lower_bound = 12 + max_k
    upper_bound = T12
    
    # Binary search
    lo = lower_bound
    hi = upper_bound
    best_solution = [T12]
    
    # We need to actually construct the solution, not just check feasibility.
    # So let's modify feasible() to return the solution if found.
    
    # Let's create a new function find_solution(rem_exp, last_f, M) that returns a list of factors or None.
    
    def find_solution(rem_exp, last_f, M):
        """
        Find a decomposition of rem_exp into distinct factors > last_f, all <= M.
        Returns a list of factors or None.
        """
        if time.time() > deadline:
            return None
            
        if rem_exp == (0, 0, 0, 0, 0):
            return []
            
        rem_val = exp_to_val(rem_exp)
        
        # If rem_val is a valid single factor
        if rem_val > last_f and rem_val >= 13 and rem_val <= M:
            return [rem_val]
            
        divs = divisor_cache[rem_exp]
        
        for d in divs:
            if d < 13:
                continue
            if d <= last_f:
                continue
            if d > M:
                break
            if d == rem_val:
                continue
            
            R = rem_val // d
            if R < d + 1:
                continue
            
            R_exp = val_to_exp(R)
            if R_exp is None:
                continue
            
            sub_sol = find_solution(R_exp, d, M)
            if sub_sol is not None:
                return [d] + sub_sol
        
        return None

    # Now binary search for the smallest M such that find_solution(INITIAL_EXP, 0, M) is not None.
    
    # To speed up, we can first check feasibility with a simpler check, then construct.
    # But find_solution already does that.
    
    # Let's iterate M from lower_bound upwards.
    # If the range is small, this is fine.
    # If not, binary search.
    
    # Let's try linear search first, as the optimal M is likely close to the lower bound.
    
    for M in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break
            
        sol = find_solution(INITIAL_EXP, 0, M)
        if sol is not None:
            # Verify
            product = 1
            for f in sol:
                product *= f
            if product == T12 and all(f > 12 for f in sol) and len(set(sol)) == len(sol) and 1 <= len(sol) <= 12:
                return sol
            # If verification fails, continue searching
            # But this shouldn't happen
            break
    
    # Fallback: return the trivial solution
    return [T12]

def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    try:
        solution = solve()
        # Verify
        product = 1
        for f in solution:
            product *= f
        if product == T12 and all(f > 12 for f in solution) and len(set(solution)) == len(solution) and 1 <= len(solution) <= 12:
            return solution
    except Exception:
        pass
    
    # Fallback
    return [T12]

# EVOLVE-BLOCK-END

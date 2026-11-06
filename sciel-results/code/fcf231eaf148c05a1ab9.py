"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    target = math.factorial(12)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exponents = [10, 5, 2, 1, 1]
    target_exp_tuple = tuple(target_exponents)
    
    # Generate all divisors of 12! that are > 12
    divisors = []
    for e0 in range(11):
        for e1 in range(6):
            for e2 in range(3):
                for e3 in range(2):
                    for e4 in range(2):
                        d = (2**e0) * (3**e1) * (5**e2) * (7**e3) * (11**e4)
                        if d > 12:
                            exps = tuple([e0, e1, e2, e3, e4])
                            divisors.append((d, exps))
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    
    n_divs = len(divisors)
    
    # Helper to calculate value from exponent tuple
    def exp_to_val(exp_tuple):
        v = 1
        for i, e in enumerate(exp_tuple):
            v *= primes[i] ** e
        return v

    @lru_cache(maxsize=None)
    def can_solve(rem_exp, max_idx):
        """
        Returns True if there is a subset of divisors[0..max_idx] 
        that multiplies to the number represented by rem_exp.
        """
        if rem_exp == (0, 0, 0, 0, 0):
            return True
        
        if max_idx < 0:
            return False
        
        # Pruning: The remaining value must be at least the smallest factor (14)
        # unless rem_exp is zero (handled above).
        # Actually, if rem_exp is not zero, we need at least one factor.
        # The smallest factor in our list is divisors[0][0] = 14.
        # So if value(rem_exp) < 14, it's impossible.
        rem_val = exp_to_val(rem_exp)
        if rem_val < divisors[0][0]:
            return False
            
        # Iterate from max_idx down to 0
        # We try to pick a divisor d_i such that d_i divides rem_val
        for i in range(max_idx, -1, -1):
            val, exps = divisors[i]
            
            # If val > rem_val, we can't use it (and since we are going down, 
            # smaller vals might work, so we continue, not break)
            if val > rem_val:
                continue
                
            # Check if exps fits in rem_exp
            valid = True
            for j in range(5):
                if exps[j] > rem_exp[j]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # If valid, recurse
            new_rem_exp = tuple(rem_exp[j] - exps[j] for j in range(5))
            if can_solve(new_rem_exp, i - 1):
                return True
                
        return False

    # Binary search for the smallest max_idx such that a solution exists
    lo = 0
    hi = n_divs - 1
    best_idx = -1
    
    # We know a solution must exist (e.g., using 12! itself if it were > 12, 
    # but 12! is huge. We need to check if any combination works).
    # The trivial solution [12!] is not allowed if 12! is not > 12? 
    # 12! is > 12. But we need distinct factors. [12!] is 1 factor.
    # Is 12! a divisor? Yes.
    # So a solution always exists: [12!].
    # Max factor is 12!.
    # We want to minimize it.
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_solve(target_exp_tuple, mid):
            best_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_idx == -1:
        # Should not happen, but fallback
        return [target]
        
    # Now extract the actual solution
    # We run a backtracking search to find one valid path using indices <= best_idx
    
    def get_solution(rem_exp, max_idx, current_factors):
        if rem_exp == (0, 0, 0, 0, 0):
            return current_factors[:]
        
        if max_idx < 0:
            return None
            
        rem_val = exp_to_val(rem_exp)
        if rem_val < divisors[0][0]:
            return None
            
        for i in range(max_idx, -1, -1):
            val, exps = divisors[i]
            
            if val > rem_val:
                continue
                
            valid = True
            for j in range(5):
                if exps[j] > rem_exp[j]:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            new_rem_exp = tuple(rem_exp[j] - exps[j] for j in range(5))
            # We don't need to check can_solve again if we trust the binary search,
            # but to be safe and fast, we can just recurse.
            # However, to avoid deep recursion or wrong paths, we can use can_solve as a guide?
            # No, simple DFS is fine.
            result = get_solution(new_rem_exp, i - 1, current_factors + [val])
            if result is not None:
                return result
                
        return None

    solution = get_solution(target_exp_tuple, best_idx, [])
    
    if solution is None:
        # Fallback if extraction failed (should not happen)
        return [target]
        
    return solution

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    target = math.factorial(12)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exponents = (10, 5, 2, 1, 1)
    primes = (2, 3, 5, 7, 11)
    
    # Generate all divisors of 12! that are > 12
    divisors = []
    for e0 in range(11):
        for e1 in range(6):
            for e2 in range(3):
                for e3 in range(2):
                    for e4 in range(2):
                        d = (2**e0) * (3**e1) * (5**e2) * (7**e3) * (11**e4)
                        if d > 12:
                            divisors.append(d)
    
    # Sort divisors
    divisors.sort()
    
    # Map divisor to exponent vector
    div_with_exp = []
    for d in divisors:
        rem = d
        exps = []
        for p in primes:
            e = 0
            while rem % p == 0:
                rem //= p
                e += 1
            exps.append(e)
        div_with_exp.append((d, tuple(exps)))
    
    n_divs = len(div_with_exp)
    vals = [d for d, _ in div_with_exp]
    exps_list = [e for _, e in div_with_exp]
    
    # Precompute for fast lookup
    # For each divisor index, store its exponent vector
    # We'll use memoization on (remaining_exps, max_idx)
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def can_solve(rem_exp, max_idx):
        """
        rem_exp: tuple of 5 exponents remaining
        max_idx: maximum index in div_with_exp we can use (inclusive)
        Returns True if a solution exists using indices 0..max_idx
        """
        if all(x == 0 for x in rem_exp):
            return True
        
        if max_idx < 0:
            return False
        
        # If the product of remaining is 1 (all zeros), handled above
        # Compute the value that would be needed if we took just one more factor
        # But that's not directly useful for the general case
        
        # Try picking divisor at index max_idx, or not picking it
        # This is a subset sum style DP
        
        # Option 1: Don't pick divisor at max_idx
        if can_solve(rem_exp, max_idx - 1):
            return True
        
        # Option 2: Pick divisor at max_idx
        exps = exps_list[max_idx]
        valid = True
        for j in range(5):
            if exps[j] > rem_exp[j]:
                valid = False
                break
        
        if valid:
            new_rem = tuple(rem_exp[j] - exps[j] for j in range(5))
            if can_solve(new_rem, max_idx - 1):
                return True
        
        return False
    
    def find_solution(rem_exp, max_idx):
        """
        Find a specific solution given that one exists.
        Returns a list of divisor indices, or None.
        """
        if all(x == 0 for x in rem_exp):
            return []
        
        if max_idx < 0:
            return None
        
        exps = exps_list[max_idx]
        valid = True
        for j in range(5):
            if exps[j] > rem_exp[j]:
                valid = False
                break
        
        if valid:
            new_rem = tuple(rem_exp[j] - exps[j] for j in range(5))
            if can_solve(new_rem, max_idx - 1):
                sub = find_solution(new_rem, max_idx - 1)
                if sub is not None:
                    return [max_idx] + sub
        
        # Try without picking max_idx
        return find_solution(rem_exp, max_idx - 1)
    
    # Iterate through possible max indices
    # We want the smallest max factor, so try increasing max_idx
    # But we need to check if a solution exists with max factor = vals[i]
    # which means using only divisors with index <= i
    
    # Clear cache before each call to be safe, but the cache is keyed by (rem_exp, max_idx)
    # so it should be fine
    
    for i in range(n_divs):
        can_solve.cache_clear()
        if can_solve(target_exponents, i):
            can_solve.cache_clear()  # Clear for find_solution
            sol_indices = find_solution(target_exponents, i)
            if sol_indices is not None:
                # Convert indices to values
                solution = [div_with_exp[idx][0] for idx in sol_indices]
                # Verify
                assert len(solution) >= 1 and len(solution) <= 12
                assert all(v > 12 for v in solution)
                assert len(set(solution)) == len(solution)
                prod = 1
                for v in solution:
                    prod *= v
                assert prod == target
                return solution
    
    # Fallback (should not happen)
    return [target]
# EVOLVE-BLOCK-END

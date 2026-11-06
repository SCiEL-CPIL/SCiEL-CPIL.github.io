"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, gcd
from functools import lru_cache

def _prime_exponents(n):
    """Return a dict of prime -> exponent for n."""
    exps = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            exps[d] = exps.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        exps[n] = exps.get(n, 0) + 1
    return exps

def _divisors_from_exponents(exp_dict):
    """Generate all divisors of a number given its prime factorization."""
    if not exp_dict:
        return [1]
    primes = list(exp_dict.keys())
    exps = [exp_dict[p] for p in primes]
    
    divisors = [1]
    for p, e in zip(primes, exps):
        current = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                current.append(d * power)
            power *= p
        divisors = current
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = factorial(12)  # 479001600
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    prime_exp_dict = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}
    
    # Generate all divisors of 12!
    all_divisors = _divisors_from_exponents(prime_exp_dict)
    
    # Filter divisors > 12
    valid_divisors = sorted([d for d in all_divisors if d > 12])
    
    # We want to find a set of distinct divisors (each > 12) whose product is N
    # Minimize the largest factor
    
    # Strategy: iterate over possible max values in ascending order
    # For each candidate max M, check if we can find distinct divisors d1, d2, ..., dk
    # such that max(di) <= M, all di > 12, product = N
    
    # Use backtracking with pruning
    # Sort valid divisors in ascending order
    
    best_solution = None
    best_max = N  # Start with the trivial solution [N]
    
    # Trivial solution
    if N > 12:
        best_solution = [N]
        best_max = N
    
    # Try to find solutions with smaller max
    # For each candidate max M, we need divisors in (12, M]
    
    # Precompute divisors grouped by value for quick lookup
    # Use a set for O(1) lookup
    divisor_set = set(valid_divisors)
    
    # For efficient search, we'll use recursive backtracking
    # Key insight: we need product of chosen divisors = N
    # So we're looking for a subset of divisors (distinct, each > 12) with product N
    
    # Better approach: work with the remaining product
    # Start with remaining = N, choose divisors from valid_divisors
    # Each chosen divisor must divide remaining, and we track used divisors
    
    # Sort divisors for consistent ordering
    sorted_valid = sorted(valid_divisors)
    
    # For each candidate max, filter divisors
    # To minimize max, iterate candidate_max in ascending order
    
    # But we need to be smart: if we fix max = M, all factors must be <= M
    # And at least one factor must be exactly M (or we could have a smaller max)
    
    # Let's iterate M in ascending order among valid divisors
    # For each M, check if there exists a set of distinct divisors from (12, M] with product N
    
    # Optimization: use memoization on (remaining_product, min_allowed_index, max_allowed_value)
    # But state space might be large
    
    # Alternative: for small M, the search space is small
    # Let's try M values in ascending order
    
    # First, find a reasonable upper bound for M
    # The trivial solution has max = N = 479001600
    # We can do better
    
    # Let's try to find solutions with small max
    # Start checking M from smallest valid divisor upward
    
    # To make this efficient, we'll use a recursive search with pruning
    # and a deadline check
    
    def search(remaining, min_idx, max_allowed, current_factors, used_set):
        """
        Find distinct divisors from sorted_valid[min_idx:] that are <= max_allowed
        with product = remaining.
        Returns list of factors or None.
        """
        if remaining == 1:
            return list(current_factors)
        
        if not current_factors and remaining > 1:
            # Need at least one more factor
            pass
        
        # Pruning: if remaining > 1, we need at least one more factor
        # Each factor must be > 12 and <= max_allowed
        # Also, each factor must divide remaining
        
        for i in range(min_idx, len(sorted_valid)):
            if time.time() > deadline:
                return None
            
            d = sorted_valid[i]
            
            if d > max_allowed:
                break
            
            if d in used_set:
                continue
            
            if remaining % d != 0:
                continue
            
            # Pruning: the product of all remaining chosen factors must be remaining // d
            # Each subsequent factor must be > d (to ensure distinctness and avoid permutations)
            # Wait, we're using min_idx to avoid permutations, but we also need distinctness
            # Since we're iterating in order and using used_set, distinctness is handled
            
            new_remaining = remaining // d
            new_factors = current_factors + [d]
            new_used = used_set | {d}
            
            # If new_remaining == 1, we're done
            if new_remaining == 1:
                return new_factors
            
            # Otherwise, continue searching
            # Next factors must be > d to avoid permutations? No, they just need to be distinct
            # But to avoid permutations, we can require factors to be non-decreasing
            # Actually, since we're picking from sorted_valid with increasing index,
            # and we track used_set, we just need to continue from i+1
            # But we also need each factor > 12, which is guaranteed by sorted_valid
            
            result = search(new_remaining, i + 1, max_allowed, new_factors, new_used)
            if result is not None:
                return result
        
        return None
    
    # Iterate candidate max values in ascending order
    # But we need to be careful: the max value must be one of the factors
    # So we iterate M over valid divisors in ascending order
    # For each M, we search for a set of distinct divisors from (12, M] with product N
    
    # To speed up, we can start from a reasonable lower bound
    # The minimum possible max is at least the 12th root of N if we could use 12 factors
    # But we can use up to 12 factors
    
    # Let's just iterate M in ascending order among valid divisors
    # But there are many divisors... let's count
    # 12! has (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792 divisors
    # That's manageable
    
    found = False
    for M in sorted_valid:
        if time.time() > deadline:
            break
        
        if M >= best_max:
            break
        
        # Search for a set of distinct divisors from valid divisors <= M with product N
        # All divisors must be > 12 (guaranteed by sorted_valid)
        # And at least one must be exactly M? Not necessarily, but if the max is < M,
        # we would have found it earlier. So for this M to be the true max,
        # we need a solution where max factor = M.
        # But it's easier to just search for any solution with all factors <= M.
        # If we find one, its actual max might be < M, but that's fine because
        # we're iterating in ascending order, so we would have found the smaller max first.
        # Wait, no: if we search with max_allowed = M and find a solution with max < M,
        # that means there's a solution with a smaller max, which we should have found
        # in an earlier iteration. So if we're at M and find a solution, its max is exactly M
        # (or we missed it earlier, which shouldn't happen).
        
        # Actually, let's just search and if we find a solution, update best_max
        result = search(N, 0, M, [], set())
        if result is not None:
            actual_max = max(result)
            if actual_max < best_max:
                best_max = actual_max
                best_solution = result
                found = True
                # We found a solution with max = actual_max <= M
                # Since we're iterating M in ascending order, this is the best we can do
                # for this M. But could there be a better solution with the same M?
                # No, we want to minimize max, and we're checking in ascending order.
                # So we can break.
                break
    
    if best_solution is None:
        best_solution = [N]
    
    return best_solution
# EVOLVE-BLOCK-END

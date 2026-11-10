"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, log
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 12
    target = factorial(N)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Primes: 2, 3, 5, 7, 11
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    
    def get_exponents(n):
        """Get prime exponent vector for n with respect to primes."""
        exps = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                exps[i] += 1
                n //= p
        return exps
    
    def add_exps(a, b):
        return tuple(x + y for x, y in zip(a, b))
    
    def sub_exps(a, b):
        return tuple(x - y for x, y in zip(a, b))
    
    def is_valid_subset(exp_tuple):
        """Check if all exponents are non-negative and don't exceed target."""
        for i, e in enumerate(exp_tuple):
            if e < 0 or e > target_exp[i]:
                return False
        return True
    
    # Generate all valid factors > 12 whose prime exponents don't exceed target
    # We need to enumerate all numbers n > 12 such that n | 12!
    # 12! = 479001600
    # Factors of 12! that are > 12
    
    # Generate all divisors of 12!
    target_tuple = tuple(target_exp)
    
    # Use recursive generation of divisors
    divisors = []
    
    def gen_divisors(idx, current_exp, current_val):
        if time.time() > deadline - 5:
            return
        if idx == len(primes):
            if current_val > N:
                divisors.append((current_val, current_exp))
            return
        p = primes[idx]
        max_e = target_exp[idx]
        val = 1
        for e in range(max_e + 1):
            gen_divisors(idx + 1, current_exp + (e,), current_val * val)
            val *= p
    
    # This recursive approach is inefficient. Let's use a different approach.
    # 12! has (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792 divisors
    # We can generate them more efficiently.
    
    divisors = []
    for a in range(11):  # 2^a, a=0..10
        for b in range(6):  # 3^b, b=0..5
            for c in range(3):  # 5^c, c=0..2
                for d in range(2):  # 7^d, d=0..1
                    for e in range(2):  # 11^e, e=0..1
                        val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d) * (11 ** e)
                        if val > N:
                            divisors.append((val, (a, b, c, d, e)))
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    
    n_div = len(divisors)
    
    # We want to find a subset of distinct divisors whose exponent sum equals target_exp
    # and minimize the maximum value in the subset.
    # Also, the number of factors must be between 1 and 12.
    
    # Binary search on the maximum factor value
    # For a given max_val M, check if there exists a valid subset with all factors <= M
    
    best_solution = None
    best_max_val = target  # 12! itself
    
    # First, try with all divisors
    # Use a smarter search: iterate over possible max values
    
    # Collect unique values
    unique_vals = sorted(set(v for v, _ in divisors))
    
    # Try different upper bounds on the maximum factor
    # We want to minimize the max, so try smaller bounds first
    
    # But we also need at least 1 factor, and the product must be exactly 12!
    # The single factor 12! works, so best_max_val starts at 12!
    
    # Let's try to find solutions with smaller max values
    # For efficiency, let's use a meet-in-the-middle or backtracking approach
    
    # Given the small size (792 divisors, but we filter to > 12), let's count
    # Actually, let's just do a smart backtracking with pruning
    
    def find_solution(max_bound, time_limit):
        """Find a valid subset with all factors <= max_bound."""
        # Filter divisors to those <= max_bound
        candidates = [(v, exp) for v, exp in divisors if v <= max_bound]
        candidates.sort(key=lambda x: -x[0])  # Sort descending for better pruning
        
        n = len(candidates)
        if n == 0:
            return None
        
        # Backtracking search
        # State: remaining exponent vector, set of used indices
        # To avoid duplicates, we enforce that we pick factors in increasing index order
        
        best = None
        found = [False]
        
        def backtrack(start_idx, remaining, current_set, current_max):
            if time.time() > time_limit:
                return
            if found[0]:
                return
            
            # Check if remaining is all zeros
            if all(r == 0 for r in remaining):
                if 1 <= len(current_set) <= 12:
                    found[0] = True
                    best_val = max(current_set) if current_set else 0
                    if best is None or best_val < best[0]:
                        best[0] = best_val
                        best[1] = list(current_set)
                return
            
            # Pruning: if any remaining component is negative, invalid
            if any(r < 0 for r in remaining):
                return
            
            # Pruning: if remaining is all positive but we can't complete with available candidates
            # Simple pruning: sum of all remaining exponents must be achievable
            
            for i in range(start_idx, n):
                if time.time() > time_limit:
                    return
                if found[0]:
                    return
                
                val, exp = candidates[i]
                
                # Check if this factor can be used
                new_remaining = sub_exps(remaining, exp)
                if not is_valid_subset(new_remaining):
                    continue
                
                # Pruning: if new_remaining has any component that can't be achieved
                # by the sum of remaining candidates' exponents, skip
                # This is expensive, so we skip this pruning for now
                
                current_set.append(val)
                backtrack(i + 1, new_remaining, current_set, max(current_max, val))
                current_set.pop()
                
                if found[0]:
                    return
        
        backtrack(0, target_exp, [], 0)
        
        return best
    
    # Try different upper bounds, starting from small values
    # We want to find the minimum possible max value
    
    # First, let's see what's achievable
    # The theoretical minimum max value: we need at least 2 factors if we want max < 12!
    # 12! = 479001600
    # sqrt(12!) ≈ 21886
    # So if we use 2 factors, the max must be at least ceil(sqrt(12!))
    
    # Let's try bounds from small to large
    # But we need to be smart about it
    
    # Try a few key bounds
    test_bounds = []
    # Add sqrt(12!) and nearby values
    import math
    sqrt_target = int(math.isqrt(target)) + 1
    test_bounds.append(sqrt_target)
    
    # Add some other reasonable bounds
    for bound in [20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 479001600]:
        if bound > N:
            test_bounds.append(bound)
    
    test_bounds = sorted(set(test_bounds))
    
    # For each bound, check if a solution exists
    # Use binary search style: start from the smallest feasible bound
    
    # Actually, let's just try to find the best solution directly
    # Start with the trivial solution
    current_best = (target, [target])
    
    # Try to improve
    # Use a more systematic approach: try bounds in increasing order
    
    for bound in test_bounds:
        if time.time() > deadline - 10:
            break
        if bound >= current_best[0]:
            break
        
        # Check if a solution exists with max <= bound
        time_limit = min(deadline - 10, time.time() + 5)
        result = find_solution(bound, time_limit)
        
        if result is not None:
            max_val, factors = result
            if max_val < current_best[0]:
                current_best = (max_val, factors)
    
    # Try to further optimize with the current best
    # Do a final search with a tighter bound
    if current_best[0] < target and time.time() < deadline - 15:
        # Try to find even better solutions
        tight_bound = current_best[0] - 1
        if tight_bound > N:
            time_limit = deadline - 5
            result = find_solution(tight_bound, time_limit)
            if result is not None:
                max_val, factors = result
                if max_val < current_best[0]:
                    current_best = (max_val, factors)
    
    # Verify the solution
    factors = current_best[1]
    product = 1
    for f in factors:
        product *= f
    
    if product != target:
        # Fallback to trivial solution
        return [target]
    
    # Check distinctness and > 12
    if len(set(factors)) != len(factors) or any(f <= N for f in factors):
        return [target]
    
    # Check count constraint
    if not (1 <= len(factors) <= 12):
        return [target]
    
    return factors
# EVOLVE-BLOCK-END

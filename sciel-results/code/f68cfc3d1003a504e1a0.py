import math
import time

T12 = math.factorial(12)  # 479001600
primes = [2, 3, 5, 7, 11]
exponents = [10, 5, 2, 1, 1]
T12_EXP = tuple(exponents)  # (10, 5, 2, 1, 1)

# Generate all divisors of 12!
def generate_divisors():
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        p_power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * p_power)
            p_power *= p
        divisors = new_divisors
    return divisors

ALL_DIVISORS = generate_divisors()

def factorize(n):
    factors = [0] * 5
    for i, p in enumerate(primes):
        while n % p == 0:
            factors[i] += 1
            n //= p
    return tuple(factors)

# Precompute (divisor, exp_tuple) for divisors > 12
DIV_DATA = []
for d in ALL_DIVISORS:
    if d > 12:
        DIV_DATA.append((d, factorize(d)))

# Sort by divisor value
DIV_DATA.sort(key=lambda x: x[0])

def solve_subset(candidates, target_exp, current_exp, start_idx, path, max_factors, best_max, deadline):
    """
    Backtracking search for a subset of candidates whose product equals 12!.
    candidates: list of (divisor, exp_tuple) sorted by divisor value
    target_exp: tuple of target prime exponents
    current_exp: tuple of current accumulated prime exponents
    start_idx: index to start from (to ensure distinctness and avoid repetition)
    path: list of selected divisors
    max_factors: remaining number of factors allowed
    best_max: current best maximum factor found (for pruning)
    deadline: time deadline
    """
    # Check if we've reached the target
    if current_exp == target_exp:
        if len(path) >= 1 and max(path) < best_max:
            return path
        return None
    
    if max_factors == 0:
        return None
    
    if time.time() > deadline:
        return None
    
    # Prune if any exponent exceeds target
    for k in range(5):
        if current_exp[k] > target_exp[k]:
            return None
    
    # Prune if we can't reach the target with remaining factors
    # The product of the remaining factors must have exponents >= remaining needed
    # This is hard to check quickly, so we rely on other pruning
    
    n = len(candidates)
    for i in range(start_idx, n):
        d, f = candidates[i]
        
        # Prune: if d >= best_max, no need to consider (we're minimizing max)
        if d >= best_max:
            break
        
        # Check if adding d keeps exponents within bounds
        new_exp = [0] * 5
        valid = True
        for k in range(5):
            ne = current_exp[k] + f[k]
            if ne > target_exp[k]:
                valid = False
                break
            new_exp[k] = ne
        
        if not valid:
            continue
        
        res = solve_subset(candidates, target_exp, tuple(new_exp), i + 1, path + [d], max_factors - 1, best_max, deadline)
        if res is not None:
            return res
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    # Try to find a solution with maximum factor as small as possible
    # Start with a large upper bound and try to reduce it
    
    # Initial upper bound: T12 itself (single factor)
    best_solution = [T12]
    best_max = T12
    
    # We'll try different upper bounds M and see if a solution exists with all factors <= M
    # Start with a reasonable M and increase if needed, or use binary search
    
    # Let's iterate M through the divisors in ascending order
    # For each M, check if T12 can be decomposed into distinct factors all <= M
    # The first feasible M gives the minimal possible largest factor
    
    # To speed up, we can try a few strategic M values first
    
    # Strategy: Try M values that are divisors of T12, in ascending order
    # But that's too many. Instead, let's try to find a good solution first.
    
    # A simple greedy approach: try to find a decomposition with small factors
    
    # Let's use the backtracking search with a decreasing best_max
    
    # First, try to find any solution with max factor < T12
    # Use the full set of candidates
    
    candidates = DIV_DATA
    
    # Try with max_factors = 12
    sol = solve_subset(candidates, T12_EXP, (0, 0, 0, 0, 0), 0, [], 12, best_max, deadline)
    
    if sol is not None:
        best_solution = sol
        best_max = max(sol)
    
    # Now try to improve by reducing the maximum factor
    # We can try different upper bounds
    
    # Iterate through candidate M values (divisors of T12) in ascending order
    # For each M, run the solver with that M as the upper bound
    
    # To be efficient, we'll try M values that are likely to be optimal
    
    # Let's try M values from the divisors list
    for d, _ in DIV_DATA:
        if d >= best_max:
            break
        if time.time() > deadline:
            break
        
        # Try to find a solution with all factors <= d
        candidates_m = [(dd, ff) for dd, ff in candidates if dd <= d]
        
        if not candidates_m:
            continue
        
        sol = solve_subset(candidates_m, T12_EXP, (0, 0, 0, 0, 0), 0, [], 12, best_max, deadline)
        
        if sol is not None:
            new_max = max(sol)
            if new_max < best_max:
                best_max = new_max
                best_solution = sol
    
    # Final check
    if best_solution and len(best_solution) > 0:
        prod = 1
        for f in best_solution:
            prod *= f
        if prod == T12 and len(set(best_solution)) == len(best_solution) and all(f > 12 for f in best_solution) and len(best_solution) <= 12:
            return best_solution
    
    return [T12]

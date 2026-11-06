import math
import time
import numpy as np
from functools import lru_cache
from scipy.optimize import milp, LinearConstraint, Bounds

# Precompute 12!
T12 = math.factorial(12)

# Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
primes = [2, 3, 5, 7, 11]
target_exponents = [10, 5, 2, 1, 1]

def generate_divisors_with_exponents():
    """
    Generate all divisors of 12! greater than 12, along with their prime factorization exponents.
    Returns: list of (divisor, [exp_2, exp_3, exp_5, exp_7, exp_11])
    """
    divisors = [(1, [0, 0, 0, 0, 0])]
    for p, e in zip(primes, target_exponents):
        new_divs = []
        p_power = 1
        for exp in range(e + 1):
            for d, exps in divisors:
                new_exps = exps.copy()
                new_exps[primes.index(p)] = exp
                new_divs.append((d * p_power, new_exps))
            p_power *= p
        divisors = new_divs
    
    # Filter: > 12, and sort by divisor
    result = sorted([(d, exps) for d, exps in divisors if d > 12], key=lambda x: x[0])
    return result

# Precompute all valid divisors with their prime factorizations
all_divisors_with_exps = generate_divisors_with_exponents()
all_divisors = [d for d, _ in all_divisors_with_exps]
divisor_set = set(all_divisors)
divisor_to_exps = {d: exps for d, exps in all_divisors_with_exps}

def solve_ilp(max_factor):
    """
    Use ILP to find if we can decompose T12 into distinct factors all <= max_factor.
    Returns a list of factors if found, None otherwise.
    """
    # Filter divisors that are <= max_factor
    valid_indices = [i for i, d in enumerate(all_divisors) if d <= max_factor]
    
    if not valid_indices:
        return None
    
    n_vars = len(valid_indices)
    valid_divs = [all_divisors[i] for i in valid_indices]
    valid_exps = [divisor_to_exps[all_divisors[i]] for i in valid_indices]
    
    # Objective: minimize sum of selected divisors (proxy for minimizing max)
    # Actually, we just need feasibility, so use zero objective
    c = np.zeros(n_vars)
    
    # Constraints:
    # 1. For each prime p_j, sum of exponents in selected divisors must equal target_exponents[j]
    # 2. At most 12 factors selected
    # 3. Each variable is binary (0 or 1)
    
    # Build constraint matrix for prime exponent constraints
    A = np.zeros((len(primes) + 1, n_vars))
    b_lower = target_exponents + [0]
    b_upper = target_exponents + [12]
    
    for j, exp_list in enumerate(valid_exps):
        for k in range(len(primes)):
            A[k, j] = exp_list[k]
        A[len(primes), j] = 1  # Count constraint
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    bounds = Bounds(0, 1)
    
    # Solve ILP
    try:
        res = milp(c=c, constraints=constraints, integrality=np.ones(n_vars), bounds=bounds)
        
        if res.status == 0 and res.success:
            # Extract selected divisors
            selected = []
            for i in range(n_vars):
                if res.x[i] > 0.5:
                    selected.append(valid_divs[i])
            
            if selected:
                # Verify the solution
                prod = 1
                for d in selected:
                    prod *= d
                if prod == T12 and len(selected) <= 12 and all(d > 12 for d in selected):
                    if len(set(selected)) == len(selected):
                        return sorted(selected)
            
            return None
        else:
            return None
    except Exception as e:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Binary search on the maximum factor T
    # Lower bound: at least 13
    # Upper bound: T12 (trivial solution)
    
    lo = 13
    hi = T12
    best_solution = [T12]  # Fallback: single factor
    
    # First, try to find a good solution with a heuristic
    # Try small number of factors first
    
    # Heuristic: try 2 factors first
    # Find divisor d of T12 such that d > 12 and T12//d > 12 and d != T12//d
    # Minimize max(d, T12//d)
    best_2factor = None
    sqrt_t12 = int(math.isqrt(T12))
    for d in range(sqrt_t12, 12, -1):
        if T12 % d == 0:
            other = T12 // d
            if d > 12 and other > 12 and d != other:
                if best_2factor is None or max(d, other) < max(best_2factor[0], best_2factor[1]):
                    best_2factor = (d, other)
                break  # First one found going down from sqrt is optimal for 2 factors
    
    if best_2factor:
        best_solution = sorted(best_2factor)
        hi = max(best_2factor)
    
    # Binary search on T using ILP
    while lo <= hi and (time.time() - start_time) < 80:
        mid = (lo + hi) // 2
        
        # Check if we can decompose T12 into distinct factors all in (12, mid]
        result = solve_ilp(mid)
        
        if result is not None:
            best_solution = sorted(result)
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Verify the best solution
    if best_solution:
        prod = 1
        for d in best_solution:
            prod *= d
        if prod == T12 and len(best_solution) <= 12 and all(d > 12 for d in best_solution):
            # Check distinctness
            if len(set(best_solution)) == len(best_solution):
                return best_solution
    
    # Fallback
    return [T12]

if __name__ == "__main__":
    res = construct()
    print(res)
    print(f"Max factor: {max(res) if res else 'N/A'}")
    print(f"Product: {math.prod(res) if res else 'N/A'}")

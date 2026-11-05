import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# Precompute primes up to 1000
PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)

def _calculate_coverage(classes, limit=10000):
    """
    Calculate the largest L such that 1..L are covered by the given classes.
    Uses a bytearray for speed.
    Returns L.
    """
    if not classes:
        return 0
    
    covered = bytearray(limit + 1)
    
    # Mark covered positions
    for p, r in classes:
        start = r if r > 0 else p
        if start < 1:
            start += p
        for n in range(start, limit + 1, p):
            covered[n] = 1
            
    # Find the largest L such that 1..L are all True
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return L

def _sum_reciprocals_exact(classes):
    """Exact Fraction sum for final verification."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def _solve_ilp_for_length(L_target, time_limit=10.0):
    """
    Solve an ILP to find a set of prime residue classes that covers 1..L_target
    with sum(1/p) <= 1.
    Returns list of [p, r] if feasible, else None.
    """
    # Variables: x[p_idx, r] for each prime p and residue r < p
    # We flatten the variables into a 1D array for scipy
    # var_index[p_idx, r] = index in the flat array
    
    num_primes = len(PRIMES)
    var_indices = {}
    flat_index = 0
    c = np.zeros(flat_index) # Placeholder, will resize
    
    # First pass: count variables
    total_vars = 0
    for p_idx, p in enumerate(PRIMES):
        total_vars += p
        
    # Second pass: assign indices
    flat_index = 0
    for p_idx, p in enumerate(PRIMES):
        for r in range(p):
            var_indices[(p_idx, r)] = flat_index
            flat_index += 1
            
    n_vars = flat_index
    
    # Objective: Minimize 0 (feasibility problem)
    # Actually, minimizing the number of classes might help find sparse solutions, 
    # but feasibility is enough. Let's use 0.
    c = np.zeros(n_vars)
    
    # Constraints:
    # 1. Distinct primes: sum_{r} x[p_idx, r] <= 1 for each p_idx
    #    This is a block diagonal constraint.
    #    We can construct A_ub and b_ub.
    
    # 2. Budget: sum_{p,r} (1/p) x[p,r] <= 1
    # 3. Coverage: For each n in 1..L_target, sum_{p,r: n% p == r} x[p,r] >= 1
    #    This is equivalent to: -sum ... <= -1
    
    # We will build the constraint matrix A (dense is too big? 
    # L_target ~ 1000, Primes ~ 168, Total vars ~ 76127 (sum of primes up to 1000 is ~76127? No, sum of primes is much smaller.
    # Sum of primes up to 1000 is roughly 1000^2 / (2 ln 1000) ~ 10^6 / 13 ~ 76000? 
    # Wait, sum of primes up to N is ~ N^2 / (2 ln N).
    # For N=1000, N^2 = 1,000,000. 2 ln 1000 ~ 13.8. So sum is ~ 72,000.
    # Total variables ~ 72,000.
    # Constraints:
    # - 168 prime constraints.
    # - 1 budget constraint.
    # - L_target coverage constraints (e.g., 1000).
    # Total constraints ~ 1200.
    # Matrix size 1200 x 72000. This is sparse. scipy.milp handles sparse.
    
    from scipy.sparse import lil_matrix, vstack, csr_matrix
    
    # Build A_ub for <= constraints
    # A_ub * x <= b_ub
    
    # Constraint 1: Distinct primes
    # For each p_idx, row has 1s at indices for that prime.
    A_prime_rows = []
    for p_idx, p in enumerate(PRIMES):
        row = np.zeros(n_vars)
        for r in range(p):
            idx = var_indices[(p_idx, r)]
            row[idx] = 1.0
        A_prime_rows.append(row)
        
    A_prime = np.array(A_prime_rows)
    b_prime = np.ones(len(PRIMES))
    
    # Constraint 2: Budget
    # sum (1/p) x <= 1
    A_budget_row = np.zeros(n_vars)
    for p_idx, p in enumerate(PRIMES):
        for r in range(p):
            idx = var_indices[(p_idx, r)]
            A_budget_row[idx] = 1.0 / p
    A_budget = A_budget_row.reshape(1, -1)
    b_budget = np.array([1.0])
    
    # Constraint 3: Coverage
    # For each n in 1..L_target: sum_{p,r: n%p==r} x[p,r] >= 1
    # => -sum ... <= -1
    A_cov_rows = []
    for n in range(1, L_target + 1):
        row = np.zeros(n_vars)
        for p_idx, p in enumerate(PRIMES):
            r = n % p
            idx = var_indices[(p_idx, r)]
            row[idx] = -1.0
        A_cov_rows.append(row)
        
    A_cov = np.array(A_cov_rows)
    b_cov = -np.ones(L_target)
    
    # Combine constraints
    A_ub = np.vstack([A_prime, A_budget, A_cov])
    b_ub = np.concatenate([b_prime, b_budget, b_cov])
    
    # Bounds: 0 <= x <= 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: all binary
    integrality = np.ones(n_vars)
    
    # Solve
    start_time = time.time()
    try:
        res = milp(c, constraints=LinearConstraint(A_ub, -np.inf, b_ub), 
                   integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit, "presolve": True})
    except Exception as e:
        return None
        
    if res.status != 0: # 0 is optimal
        return None
        
    # Extract solution
    x = res.x
    classes = []
    for p_idx, p in enumerate(PRIMES):
        for r in range(p):
            idx = var_indices[(p_idx, r)]
            if x[idx] > 0.5:
                classes.append([p, r])
                
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Get an initial feasible solution using a simple greedy method to set lower bound
    # Simple greedy: cover 1..100
    initial_classes = []
    used_primes = set()
    budget = 0.0
    covered = [False] * 200
    for n in range(1, 200):
        if covered[n]: continue
        # Find smallest prime not used that fits budget
        for p in PRIMES:
            if p in used_primes: continue
            if budget + 1.0/p <= 1.0:
                r = n % p
                initial_classes.append([p, r])
                used_primes.add(p)
                budget += 1.0/p
                start = r if r > 0 else p
                if start < 1: start += p
                for k in range(start, 200, p):
                    covered[k] = True
                break
        if time.time() > deadline - 130: break
        
    if not initial_classes:
        # Fallback to a known small solution
        return [[2, 1], [3, 2], [5, 4]]
        
    current_best_L = _calculate_coverage(initial_classes, limit=5000)
    best_classes = initial_classes
    
    # 2. Binary search for larger L using ILP
    # We want to find max L such that ILP is feasible.
    # Start from current_best_L + 1
    # Upper bound: Theoretically, with budget 1, we can cover quite a lot.
    # Let's try to go up to 3000 or 5000 if time permits.
    
    low = current_best_L
    high = 5000 # Arbitrary upper bound for search
    
    # We will try to increase L in chunks or binary search.
    # Binary search is efficient if ILP solve time is consistent.
    # However, ILP difficulty increases with L.
    
    # Let's do a linear search with steps, or binary search.
    # Binary search on feasibility.
    
    last_feasible_L = current_best_L
    last_feasible_classes = best_classes
    
    # To speed up, we can try to solve for L = current_best_L + 100, then + 200, etc?
    # Or just binary search between low and high.
    
    while low < high and time.time() < deadline - 20.0:
        mid = (low + high + 1) // 2
        
        # Solve ILP for mid
        # Limit time per solve to avoid getting stuck
        time_per_solve = 10.0
        if time.time() + time_per_solve > deadline - 10.0:
            break
            
        classes = _solve_ilp_for_length(mid, time_limit=time_per_solve)
        
        if classes is not None:
            # Verify coverage and budget
            if _sum_reciprocals_exact(classes) <= 1:
                L_check = _calculate_coverage(classes, limit=10000)
                if L_check >= mid:
                    # Feasible
                    last_feasible_L = mid
                    last_feasible_classes = classes
                    low = mid
                else:
                    # ILP said feasible but coverage check failed? Should not happen if constraints are correct.
                    # Maybe numerical issues. Treat as infeasible to be safe.
                    high = mid - 1
            else:
                # Budget violation due to float issues in ILP?
                # Try to prune? Or treat as infeasible.
                high = mid - 1
        else:
            # Infeasible
            high = mid - 1
            
    # 3. Local Search Refinement on the best ILP solution
    # The ILP solution might be "sparse" or suboptimal for extending L further.
    # We can use the ILP solution as a start for local search.
    
    current_classes = [c[:] for c in last_feasible_classes]
    current_L = _calculate_coverage(current_classes, limit=10000)
    
    # Simple local search: try to fix gaps
    improvement_found = True
    while improvement_found and time.time() < deadline - 5.0:
        improvement_found = False
        
        # Find first gap
        check_limit = current_L + 100
        if check_limit > 10000: check_limit = 10000
        
        covered = bytearray(check_limit + 1)
        for p, r in current_classes:
            start = r if r > 0 else p
            if start < 1: start += p
            for n in range(start, check_limit + 1, p):
                covered[n] = 1
                
        gap = -1
        for i in range(1, check_limit + 1):
            if not covered[i]:
                gap = i
                break
                
        if gap == -1:
            if check_limit > current_L:
                current_L = check_limit
                last_feasible_L = current_L
                last_feasible_classes = current_classes
                improvement_found = True
            continue
            
        # Try to fix gap
        best_new_L = current_L
        best_new_classes = None
        
        # Option 1: Change residue of existing prime
        for idx, (p, r) in enumerate(current_classes):
            new_r = gap % p
            if new_r == r: continue
            trial = [c[:] for c in current_classes]
            trial[idx] = [p, new_r]
            L_trial = _calculate_coverage(trial, limit=current_L + 100)
            if L_trial > best_new_L:
                best_new_L = L_trial
                best_new_classes = trial
                
        # Option 2: Add new prime
        used_primes = set(p for p, r in current_classes)
        budget = sum(1.0/p for p, r in current_classes)
        for p in PRIMES:
            if p in used_primes: continue
            if budget + 1.0/p <= 1.0:
                new_r = gap % p
                trial = [c[:] for c in current_classes]
                trial.append([p, new_r])
                L_trial = _calculate_coverage(trial, limit=current_L + 100)
                if L_trial > best_new_L:
                    best_new_L = L_trial
                    best_new_classes = trial
                break # Try smallest prime first
                
        if best_new_classes is not None and best_new_L > current_L:
            current_classes = best_new_classes
            current_L = best_new_L
            last_feasible_L = current_L
            last_feasible_classes = current_classes
            improvement_found = True
            
    # Final check
    if _sum_reciprocals_exact(last_feasible_classes) <= 1:
        return last_feasible_classes
    else:
        return [[2, 1], [3, 2], [5, 4]]

# EVOLVE-BLOCK-END

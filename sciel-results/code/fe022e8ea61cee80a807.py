import time
import math
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def _is_valid_full(S):
    """Full validity check for a set S."""
    if not S:
        return True
    # Compute all nonempty subset sums
    sums = set()
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            pass # Placeholder, using iterative method below for correctness
    # Efficient subset sum generation
    current_sums = set()
    for x in S:
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        new_sums.add(x)
        current_sums.update(new_sums)
    
    sums_list = sorted(current_sums)
    total = sum(S)
    
    # Check divisibility
    for s in sums_list:
        if s == 0:
            continue
        # Check multiples
        multiple = 2 * s
        while multiple <= total:
            if multiple in current_sums:
                return False
            multiple += s
        # Check divisors
        for d in range(1, int(math.isqrt(s)) + 1):
            if s % d == 0:
                if d != s and d in current_sums:
                    return False
                other = s // d
                if other != s and other != d and other in current_sums:
                    return False
    return True

def _greedy_fallback(pool, deadline):
    """Fast greedy construction to ensure a valid result if ILP fails or times out."""
    current = []
    sums_set = set()
    total = 0
    
    # Try larger numbers first, they have fewer multiples
    candidates = sorted(pool, reverse=True)
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        new_sums = set(sums_set)
        for s in sums_set:
            new_sums.add(s + x)
        new_sums.add(x)
        new_total = total + x
        
        # Quick check: no new sum divides or is divided by an existing sum
        valid = True
        for ns in new_sums:
            for os in sums_set:
                if ns == os:
                    continue
                if ns % os == 0 or os % ns == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check new sums against each other
            valid2 = True
            ns_list = sorted(new_sums)
            for i in range(len(ns_list)):
                for j in range(i+1, len(ns_list)):
                    if ns_list[j] % ns_list[i] == 0:
                        valid2 = False
                        break
                if not valid2:
                    break
            
            if valid2:
                current.append(x)
                sums_set = new_sums
                total = new_total
                if len(current) >= 30:
                    break
    return current

def construct():
    start_time = time.time()
    # Safety margin: return 5s before 90s limit
    deadline = start_time + 85
    
    pool = list(range(1, 257))
    
    # Try ILP approach
    ilp_success = False
    best_set = []
    
    # We will try to maximize the number of selected elements.
    # Since ILP might be slow or fail to prove optimality quickly, we run it with a time limit.
    # However, scipy.milp doesn't have a direct "time limit" parameter in the standard interface 
    # that guarantees early return with a feasible solution easily accessible in all versions.
    # Instead, we will construct the ILP for a specific target size K and binary search for the max K.
    # Or, better, we maximize sum(x_i) with objective coefficients 1 for each x_i.
    
    n = len(pool)
    c = np.ones(n)
    
    # Constraints:
    # 1. Subset sum divisibility.
    # For any subset A and B (disjoint? No, any two subsets with sums s1, s2), s1 does not divide s2.
    # This is hard to linearize directly for ALL subsets.
    # However, we can restrict the search space.
    # A key insight: if we pick numbers that are "large" and "incomparable", conflicts are rare.
    # The ILP formulation for "no two subset sums divide each other" is extremely dense.
    # Number of subsets is 2^256. We cannot list all constraints.
    
    # Therefore, a full ILP on 256 variables with all subset-sum constraints is intractable.
    # We must restrict the pool or use a heuristic ILP on a smaller, promising subset.
    
    # Strategy: 
    # 1. Identify a smaller pool of "promising" candidates (e.g., top 50 largest numbers, or numbers with few divisors).
    # 2. Run ILP on this smaller pool to find a valid set.
    # 3. Use the result to seed a local search on the full pool.
    
    # Let's select a pool of 40 candidates.
    # Heuristic: Numbers with few divisors (primes or semiprimes) are better.
    # Also, larger numbers are better because their multiples are larger (less likely to be in the set).
    
    def divisor_count(x):
        if x < 2: return 1
        count = 2
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                count += 2
        return count

    # Sort by (divisor_count, -value) -> Prefer few divisors, then large value
    ranked_pool = sorted(pool, key=lambda x: (divisor_count(x), -x))
    ilp_pool = ranked_pool[:40]
    
    n_ilp = len(ilp_pool)
    
    # Build ILP constraints for this smaller pool
    # Variables: z_i in {0,1} for i in 0..n_ilp-1
    
    # Objective: maximize sum(z_i)
    c_ilp = np.ones(n_ilp)
    
    # Constraints:
    # We need to ensure that for any two distinct non-empty subsets S1, S2 of the selected elements,
    # sum(S1) does not divide sum(S2).
    # This is still 2^40 subsets. Too many.
    
    # Alternative: Use a "conflict graph" approach?
    # If we add element x, it creates new sums.
    # This is essentially what the greedy does.
    
    # Let's try a different ILP formulation:
    # Maximize sum(z_i)
    # Subject to: 
    # For every pair of elements i, j? No, conflicts involve sums of multiple elements.
    
    # Given the difficulty of exact ILP for this constraint, let's use ILP to select a subset 
    # that maximizes a "quality score" based on pairwise non-divisibility of the elements themselves,
    # and then verify/extend.
    # Actually, the condition is on SUBSET SUMS.
    
    # Let's stick to a robust heuristic that leverages the "ILP-like" structure of checking constraints,
    # but implemented efficiently. The prompt asks for a DIFFERENT strategy.
    # The current strategy is Greedy + Local Search + Backtracking.
    
    # New Strategy: **Simulated Annealing** on the set of selected elements.
    # State: A subset of the pool.
    # Move: Add, remove, or swap an element.
    # Cost: Number of divisibility violations in the subset sums.
    # We want to reach Cost = 0 with maximum size.
    
    # This is a significant structural change from deterministic backtracking/greedy.
    
    def get_violations(S):
        """Return number of divisibility violations and the set itself."""
        if not S:
            return 0, set()
        sums = set()
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            sums.update(new_sums)
        
        sums_list = sorted(sums)
        total = sum(S)
        violations = 0
        # Optimization: Only check if a multiple exists in the set
        sum_set = sums
        for s in sums_list:
            if s == 0: continue
            m = 2 * s
            while m <= total:
                if m in sum_set:
                    violations += 1
                m += s
            # Divisors
            for d in range(1, int(math.isqrt(s)) + 1):
                if s % d == 0:
                    if d != s and d in sum_set:
                        violations += 1
                    other = s // d
                    if other != s and other != d and other in sum_set:
                        violations += 1
        return violations, sums
    
    # Initialize with a large random set or greedy set
    rng = random.Random(42)
    current_set = _greedy_fallback(pool, deadline - 5)
    if not current_set:
        current_set = [pool[-1]]
    
    current_violations, current_sums = get_violations(current_set)
    best_set = current_set[:]
    best_violations = current_violations
    best_size = len(current_set)
    
    T = 10.0
    T_min = 0.01
    alpha = 0.99
    
    while time.time() < deadline - 2:
        if T < T_min:
            break
        
        # Generate a neighbor
        neighbor = current_set[:]
        move_type = rng.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Add a random element not in set
            candidates = [x for x in pool if x not in neighbor]
            if candidates:
                x = rng.choice(candidates)
                neighbor.append(x)
        elif move_type == 'remove':
            if neighbor:
                idx = rng.randrange(len(neighbor))
                neighbor.pop(idx)
        else: # swap
            if neighbor:
                candidates = [x for x in pool if x not in neighbor]
                if candidates:
                    x_new = rng.choice(candidates)
                    idx = rng.randrange(len(neighbor))
                    x_old = neighbor[idx]
                    neighbor[idx] = x_new
        
        new_violations, _ = get_violations(neighbor)
        
        # Acceptance criterion
        # We prefer larger sets with fewer violations.
        # Score = -size * 100 + violations ?
        # Let's use: Energy = violations - 0.1 * size
        # We want to minimize Energy.
        
        old_energy = current_violations - 0.1 * len(current_set)
        new_energy = new_violations - 0.1 * len(neighbor)
        delta_E = new_energy - old_energy
        
        if delta_E < 0 or (T > 0 and rng.random() < math.exp(-delta_E / T)):
            current_set = neighbor
            current_violations = new_violations
            
            if current_violations == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
        
        T *= alpha
    
    # Final check and cleanup
    if best_violations > 0:
        # If we didn't find a valid set, fallback to greedy
        fallback = _greedy_fallback(pool, deadline)
        if len(fallback) > len(best_set) or (len(fallback) == len(best_set) and best_violations > 0):
             best_set = fallback
             best_violations, _ = get_violations(best_set)
    
    # Ensure validity
    if best_set and not _is_valid_full(best_set):
        # Last resort: empty set or single element
        if pool:
            best_set = [pool[-1]]
        else:
            best_set = []
            
    return [int(x) for x in best_set]

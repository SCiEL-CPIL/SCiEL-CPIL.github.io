import math
import numpy as np
import time

def construct():
    n = 256
    deadline = time.time() + 80  # 80s deadline with 10s safety margin
    
    # Precompute prime factors for each number
    def prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    factors_cache = {i: prime_factors(i) for i in range(1, n + 1)}
    
    def is_coprime(a, b):
        # Check if sets of prime factors are disjoint
        return len(factors_cache[a] & factors_cache[b]) == 0
    
    # Try MILP first with a time limit
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables x_1 ... x_256
        c = -np.arange(1, n + 1, dtype=float)
        
        # Identify conflicting pairs
        conflicts = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if math.gcd(i, j) > 1:
                    conflicts.append((i, j))
        
        num_vars = n
        num_constraints = len(conflicts)
        
        A = np.zeros((num_constraints, num_vars))
        b_lower = np.zeros(num_constraints)
        b_upper = np.ones(num_constraints)
        
        for k, (i, j) in enumerate(conflicts):
            A[k, i - 1] = 1
            A[k, j - 1] = 1
            
        linear_constraint = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))
        integrality = np.ones(num_vars)
        
        result = milp(c, constraints=linear_constraint, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            sol = result.x
            selected = []
            for i in range(num_vars):
                if sol[i] > 0.5:
                    selected.append(i + 1)
            if 1 not in selected:
                selected.append(1)
            # Validate
            valid = True
            for idx in range(len(selected)):
                for jdx in range(idx + 1, len(selected)):
                    if math.gcd(selected[idx], selected[jdx]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return [int(x) for x in selected]
                
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        # Not enough time, return a quick greedy result
        selected = [1]
        for i in range(n, 1, -1):
            if all(math.gcd(i, s) == 1 for s in selected):
                selected.append(i)
        return [int(x) for x in selected]
    
    # Greedy + local search approach
    # Start with greedy: sort by value descending, add if coprime to all
    selected = []
    for i in range(n, 0, -1):
        if all(math.gcd(i, s) == 1 for s in selected):
            selected.append(i)
    
    current_sum = sum(selected)
    selected_set = set(selected)
    
    # Local search: try swapping one element for another to improve
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try removing each element and trying to add a better one
        for idx, elem in enumerate(list(selected_set)):
            if time.time() > deadline - 2:
                break
            # Remove elem
            remaining = [x for x in selected_set if x != elem]
            # Try adding each number not in selected_set that is coprime to all remaining
            best_add = -1
            best_gain = 0
            for candidate in range(n, 0, -1):
                if candidate in selected_set:
                    continue
                if candidate == elem:
                    continue
                # Check if candidate is coprime to all in remaining
                if all(math.gcd(candidate, r) == 1 for r in remaining):
                    gain = candidate - elem
                    if gain > best_gain:
                        best_gain = gain
                        best_add = candidate
            if best_add > 0:
                selected_set.remove(elem)
                selected_set.add(best_add)
                improved = True
                current_sum += best_gain
    
    # Try adding 1 if not present
    if 1 not in selected_set:
        if all(math.gcd(1, s) == 1 for s in selected_set):
            selected_set.add(1)
    
    result = sorted(selected_set)
    # Final validation
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Shouldn't happen, but return greedy if it does
                selected = [1]
                for i2 in range(n, 1, -1):
                    if all(math.gcd(i2, s) == 1 for s in selected):
                        selected.append(i2)
                return [int(x) for x in selected]
    
    return [int(x) for x in result]

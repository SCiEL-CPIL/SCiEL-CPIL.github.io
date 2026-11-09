def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    from math import gcd
    import numpy as np
    
    N = 256
    
    # Find all primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(N)
    
    # Build constraint matrix: for each prime p, sum of x_n over n divisible by p <= 1
    # Number of constraints = len(primes)
    num_constraints = len(primes)
    
    # Build sparse representation of A matrix
    # A[i][j] = 1 if (j+1) is divisible by primes[i], else 0
    # Using lists for scipy sparse
    row_indices = []
    col_indices = []
    data = []
    
    for i, p in enumerate(primes):
        for n in range(p, N + 1, p):
            row_indices.append(i)
            col_indices.append(n - 1)  # 0-indexed
            data.append(1.0)
    
    from scipy.sparse import csc_matrix
    A = csc_matrix((data, (row_indices, col_indices)), shape=(num_constraints, N))
    
    b = np.ones(num_constraints)
    
    # Objective: maximize sum(n * x_n) => minimize -sum(n * x_n)
    c = -np.arange(1, N + 1, dtype=float)
    
    bounds = [(0, 1)] * N
    integrality = np.ones(N)
    
    try:
        from scipy.optimize import milp
        res = milp(
            c,
            constraints=(A, b),
            bounds=bounds,
            integrality=integrality,
            options={'time_limit': 45, 'mip_rel_gap': 1e-6}
        )
        
        if res.status == 0:  # Optimal
            x = res.x
            result = [int(i + 1) for i, val in enumerate(x) if val > 0.5]
            if 1 not in result:
                result.append(1)
            result = [int(v) for v in result]  # Ensure Python int
            return result
        elif res.status == 1:  # Locally optimal
            x = res.x
            result = [int(i + 1) for i, val in enumerate(x) if val > 0.5]
            if 1 not in result:
                result.append(1)
            result = [int(v) for v in result]
            return result
        elif res.status == 2:  # Feasible
            x = res.x
            result = [int(i + 1) for i, val in enumerate(x) if val > 0.5]
            if 1 not in result:
                result.append(1)
            result = [int(v) for v in result]
            return result
            
    except Exception:
        pass
    
    # Fallback: greedy + local search
    if time.time() > deadline - 5:
        # Very fast fallback: prime powers + 1
        result = [1]
        for p in primes:
            max_p = p
            while max_p * p <= N:
                max_p *= p
            result.append(max_p)
        return [int(v) for v in result]
    
    # Greedy: sort by value descending, add if no conflict
    selected = set()
    # Start with 1
    selected.add(1)
    
    # For greedy, process numbers in descending order
    for n in range(N, 0, -1):
        if n in selected:
            continue
        # Check if n is coprime to all selected
        compatible = True
        for s in selected:
            if gcd(n, s) > 1:
                compatible = False
                break
        if compatible:
            selected.add(n)
    
    # Local search: try swaps and additions
    def get_conflict(n, sel):
        """Return set of selected numbers that conflict with n"""
        return {s for s in sel if gcd(n, s) > 1}
    
    best_sum = sum(selected)
    no_improve = 0
    max_iter = 50000
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        if no_improve > 5000:
            break
            
        improved = False
        
        # Try adding a number
        for n in range(1, N + 1):
            if n in selected:
                continue
            conflicts = get_conflict(n, selected)
            if not conflicts:
                # Can add n
                selected.add(n)
                new_sum = best_sum + n
                if new_sum > best_sum:
                    best_sum = new_sum
                    improved = True
                    no_improve = 0
                    break
            elif len(conflicts) == 1:
                # Swap: remove the one conflict, add n
                conflict = next(iter(conflicts))
                if n > conflict:
                    selected.remove(conflict)
                    selected.add(n)
                    new_sum = best_sum - conflict + n
                    if new_sum > best_sum:
                        best_sum = new_sum
                        improved = True
                        no_improve = 0
                        break
                    else:
                        selected.remove(n)
                        selected.add(conflict)
            
        if not improved:
            no_improve += 1
    
    result = list(selected)
    if 1 not in result:
        result.append(1)
    return [int(v) for v in result]

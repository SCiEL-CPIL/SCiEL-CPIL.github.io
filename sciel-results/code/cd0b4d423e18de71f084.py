import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Precompute variable indices for each (p, r)
    var_idx = {}
    var_count = 0
    for p in primes:
        for r in range(p):
            var_idx[(p, r)] = var_count
            var_count += 1
            
    n_vars = var_count
    
    # Precompute prime indices for faster access in constraint building
    # We need to know which variables correspond to which prime for distinct constraint
    # And which variables cover which n for coverage constraint.
    
    # For distinct constraint: for each p, list of var indices
    prime_vars = {p: [var_idx[(p, r)] for r in range(p)] for p in primes}
    
    # For coverage constraint: for each n, list of (p, r) that cover n?
    # Or for each (p, r), which n does it cover?
    # It's easier to iterate n, then p, then r = n % p.
    
    def build_and_solve(L):
        """Build ILP for covering [1, L] and solve. Returns x vector or None."""
        c = np.zeros(n_vars)
        
        row_indices = []
        col_indices = []
        data = []
        
        current_row = 0
        
        # 1. Distinct prime constraints: sum_{r} x_{p,r} <= 1
        for p in primes:
            for idx in prime_vars[p]:
                row_indices.append(current_row)
                col_indices.append(idx)
                data.append(1)
            current_row += 1
            
        # 2. Budget constraint: sum (1/p) x_{p,r} <= 1
        for p in primes:
            inv_p = 1.0 / p
            for idx in prime_vars[p]:
                row_indices.append(current_row)
                col_indices.append(idx)
                data.append(inv_p)
        current_row += 1
        
        # 3. Coverage constraints: for n in 1..L, sum_{p,r: n%p==r} x_{p,r} >= 1
        for n in range(1, L + 1):
            for p in primes:
                r = n % p
                idx = var_idx[(p, r)]
                row_indices.append(current_row)
                col_indices.append(idx)
                data.append(1)
            current_row += 1
            
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(current_row, n_vars))
        
        lb = np.zeros(current_row)
        ub = np.full(current_row, np.inf)
        
        # Distinct primes: 0 <= sum <= 1
        ub[:len(primes)] = 1
        
        # Budget: 0 <= sum <= 1
        ub[len(primes)] = 1
        
        # Coverage: 1 <= sum <= inf
        lb[len(primes)+1:] = 1
        
        constraints = LinearConstraint(A, lb, ub)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraints, integrality=np.ones(n_vars), bounds=bounds)
            if res.status == 0:
                return res.x
            else:
                return None
        except Exception:
            return None

    # Binary search for max L
    low = 1
    high = 2000
    
    # Check high
    if build_and_solve(high) is not None:
        # Try higher
        low = high
        high = 5000
        # Check 5000
        if build_and_solve(high) is not None:
            best_L = high
        else:
            while low < high:
                mid = (low + high + 1) // 2
                if time.time() > deadline - 10:
                    break
                if build_and_solve(mid) is not None:
                    low = mid
                else:
                    high = mid - 1
            best_L = low
    else:
        # Binary search in [1, high]
        while low < high:
            mid = (low + high + 1) // 2
            if time.time() > deadline - 10:
                break
            if build_and_solve(mid) is not None:
                low = mid
            else:
                high = mid - 1
        best_L = low
        
    # Retrieve solution for best_L
    x_sol = build_and_solve(best_L)
    
    if x_sol is None:
        # Fallback
        return []
        
    classes = []
    for (p, r), idx in var_idx.items():
        if x_sol[idx] > 0.5:
            classes.append([p, r])
            
    return classes

import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from fractions import Fraction

_PRIMES_CACHE = None

def _get_primes():
    global _PRIMES_CACHE
    if _PRIMES_CACHE is None:
        primes = []
        for p in range(2, 1001):
            is_prime = True
            for q in range(2, int(p**0.5) + 1):
                if p % q == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(p)
        _PRIMES_CACHE = primes
    return _PRIMES_CACHE

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    primes = _get_primes()
    exact_recips = {p: Fraction(1, p) for p in primes}
    
    best_solution = None
    best_L = 0
    
    # We will try L from 1000 downwards.
    # To avoid rebuilding the full ILP for every L, we can just try a few key Ls.
    # Or, we can solve for L=1000. If it fails, try 990, 980...
    
    for L in range(1000, 0, -5): # Step 5 to save time
        if time.time() > deadline - 15.0:
            break
            
        # Filter primes: Only p <= L is strictly necessary for efficiency, 
        # but p > L covers 1 number. Let's include all p <= 1000 for now, 
        # but maybe restrict to p <= L * 2? 
        # Actually, for L=1000, all primes are <= 1000.
        
        # Build sparse ILP
        # Variables: (i, r) for p_i in primes
        # To reduce size, let's only consider primes p <= L.
        # If p > L, it covers n=r. 
        # We can handle "holes" later or assume optimal solutions use p <= L.
        # Let's try with all primes first, but if it's too slow, we restrict.
        
        active_primes = [p for p in primes if p <= L]
        if not active_primes:
            continue
            
        # Map (i, r) to flat index
        idx_map = {}
        rows = []
        cols = []
        data = []
        
        # 1. Prime Uniqueness Constraints
        # For each prime i, sum_r x_{i,r} <= 1
        var_idx = 0
        prime_vars = [] # list of lists of var indices for each prime
        for i, p in enumerate(active_primes):
            start = var_idx
            for r in range(p):
                idx_map[(i, r)] = var_idx
                var_idx += 1
            end = var_idx
            prime_vars.append((start, end))
            
        N = var_idx
        
        # Constraint 1: Uniqueness
        for start, end in prime_vars:
            for k in range(start, end):
                rows.append(len(rows)) # Row index
                cols.append(k)
                data.append(1.0)
        # This logic is getting messy for sparse construction manually.
        # Let's use coo format.
        
        coo_rows = []
        coo_cols = []
        coo_data = []
        
        # 1. Uniqueness
        for start, end in prime_vars:
            row_idx = len(coo_rows) // (end - start) if coo_rows else 0 # Wrong
            # Better:
            pass
            
        # Let's just build dense if N is small enough?
        # N for L=1000 is 76k. Dense 1000x76k is 76M floats. 600MB. Too big.
        # Must use sparse.
        
        # Let's use a different approach: 
        # Iterate through constraints and add to COO lists.
        
        # Reset
        coo_rows = []
        coo_cols = []
        coo_data = []
        
        num_constraints = 0
        
        # 1. Uniqueness constraints (one per prime)
        for start, end in prime_vars:
            for k in range(start, end):
                coo_rows.append(num_constraints)
                coo_cols.append(k)
                coo_data.append(1.0)
            num_constraints += 1
            
        # 2. Budget constraint
        for i, p in enumerate(active_primes):
            start, end = prime_vars[i]
            for k in range(start, end):
                coo_rows.append(num_constraints)
                coo_cols.append(k)
                coo_data.append(1.0 / p)
        num_constraints += 1
        
        # 3. Coverage constraints (one per n in 1..L)
        for n in range(1, L + 1):
            for i, p in enumerate(active_primes):
                r = n % p
                k = idx_map[(i, r)]
                coo_rows.append(num_constraints)
                coo_cols.append(k)
                coo_data.append(1.0)
            num_constraints += 1
            
        A = csr_matrix((coo_data, (coo_rows, coo_cols)), shape=(num_constraints, N))
        
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)
        
        # Set bounds
        # 1..K primes: 0 <= sum <= 1
        for i in range(len(active_primes)):
            lb[i] = 0
            ub[i] = 1
        # Budget: 0 <= sum <= 1
        lb[len(active_primes)] = 0
        ub[len(active_primes)] = 1.0 - 1e-7
        # Coverage: 1 <= sum <= inf
        for i in range(len(active_primes) + 1, num_constraints):
            lb[i] = 1
            ub[i] = np.inf
            
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(N)
        bounds = Bounds(0, 1)
        c = np.zeros(N)
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                x = res.x
                classes = []
                budget_sum = Fraction(0, 1)
                
                for i, p in enumerate(active_primes):
                    for r in range(p):
                        if x[idx_map[(i, r)]] > 0.5:
                            classes.append([p, r])
                            budget_sum += exact_recips[p]
                            break
                            
                if budget_sum <= 1:
                    # Verify coverage
                    covered = [False] * (L + 1)
                    for p, r in classes:
                        for n in range(r, L + 1, p):
                            if n >= 1:
                                covered[n] = True
                    if all(covered[1:L+1]):
                        best_solution = classes
                        best_L = L
                        break
                        
        except Exception:
            pass

    if best_solution is None:
        primes_list = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_list)]
        
    return best_solution

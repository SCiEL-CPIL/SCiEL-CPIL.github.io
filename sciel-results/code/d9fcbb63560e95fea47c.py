# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def _primes_up_to(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def construct():
    primes = _primes_up_to(1000)
    num_primes = len(primes)
    
    # Pre-calculate variable indices
    var_map = {}
    var_idx = 0
    for i, p in enumerate(primes):
        for r in range(p):
            var_map[(p, r)] = var_idx
            var_idx += 1
            
    total_vars = var_idx
    
    # Objective: Minimize 0 (Feasibility)
    c = np.zeros(total_vars)
    
    # Bounds
    integrality = np.ones(total_vars)
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    def solve_ilp(L):
        if L <= 0:
            return None
            
        # Build Coverage Constraints
        rows = []
        cols = []
        data = []
        
        for n in range(1, L + 1):
            for p in primes:
                r = n % p
                v_idx = var_map[(p, r)]
                rows.append(n - 1)
                cols.append(v_idx)
                data.append(1)
                
        A_cov = csr_matrix((data, (rows, cols)), shape=(L, total_vars))
        
        # Budget Constraint
        b_data = []
        b_cols = []
        for p in primes:
            cost = 1.0 / p
            for r in range(p):
                v_idx = var_map[(p, r)]
                b_data.append(cost)
                b_cols.append(v_idx)
                
        budget_rows = np.zeros(len(b_cols), dtype=int)
        A_budget = csr_matrix((b_data, (budget_rows, b_cols)), shape=(1, total_vars))
        
        # Distinct Prime Constraints
        d_rows = []
        d_cols = []
        d_data = []
        for i, p in enumerate(primes):
            for r in range(p):
                v_idx = var_map[(p, r)]
                d_rows.append(i)
                d_cols.append(v_idx)
                d_data.append(1)
                
        A_distinct = csr_matrix((d_data, (d_rows, d_cols)), shape=(num_primes, total_vars))
        
        # Combine Constraints
        A_all = np.vstack([
            A_cov,
            A_budget,
            A_distinct
        ])
        
        # LinearConstraint
        # Coverage: >= 1 (lb=1, ub=inf)
        # Budget: <= 1 (lb=0, ub=1)
        # Distinct: <= 1 (lb=0, ub=1)
        
        lb_vec = np.concatenate([
            np.ones(L),
            np.zeros(1),
            np.zeros(num_primes)
        ])
        
        ub_vec = np.concatenate([
            np.full(L, np.inf),
            np.ones(1),
            np.ones(num_primes)
        ])
        
        cons = LinearConstraint(A_all, lb=lb_vec, ub=ub_vec)
        
        res = milp(
            c=c,
            constraints=cons,
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0:
            return res.x
        else:
            return None

    # Binary Search for L
    low = 1
    high = 2000
    
    # Ensure high is not feasible, find a feasible high
    # Actually, if high is feasible, we might want to go higher?
    # But primes are limited to 1000.
    # Max L is bounded.
    # Let's just check if high is feasible. If yes, maybe increase high?
    # But 2000 is a good guess. If 2000 is feasible, we can try 5000?
    # The problem says "capped at 10^5".
    # Let's try up to 10000 to be safe, but start binary search there.
    
    # Check 10000
    if solve_ilp(10000) is not None:
         high = 10000
    else:
         # Find a feasible high
         while high > low and solve_ilp(high) is None:
             high = high // 2
             if high < 1:
                 return []
                 
    best_L = low
    best_sol = None
    
    while low <= high:
        mid = (low + high) // 2
        sol = solve_ilp(mid)
        if sol is not None:
            best_L = mid
            best_sol = sol
            low = mid + 1
        else:
            high = mid - 1
            
    if best_sol is None:
        return []
        
    classes = []
    for i, p in enumerate(primes):
        for r in range(p):
            v_idx = var_map[(p, r)]
            if best_sol[v_idx] > 0.5:
                classes.append([p, r])
                
    return classes

# EVOLVE-BLOCK-END

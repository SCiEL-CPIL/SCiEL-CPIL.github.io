import numpy as np
import time
from scipy.optimize import linprog

def _compute_mobius(K):
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0 - 15.0  # safety margin
    
    # Strategy: Use LP with a moderate set of k values, then verify M(x) constraints
    # The key insight: M(x) = sum_k f(k) * floor(x/k) must be <= 1.0001 for all x in [1, 10K]
    # where K is the maximum key in the domain.
    
    # We'll try different sizes of K_max and find the best feasible solution
    # The objective is to maximize a = -sum_k f(k) * log(k) / k
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try various K_max values
    for K_max in [20, 30, 50, 80, 120, 200]:
        if time.time() > deadline:
            break
            
        ks = np.arange(2, K_max + 1)
        m = len(ks)
        
        # Objective: maximize -sum f_k log(k)/k => minimize sum f_k log(k)/k
        c_obj = np.log(ks) / ks
        
        # Build constraint matrix for M(x) <= 1
        # M(x) = sum_k f(k) * floor(x/k) <= 1
        # This is a linear constraint in f
        max_x = 10 * K_max
        
        # Use a subset of x values to keep the LP manageable
        # The critical x values are those where floor(x/k) changes, i.e., x = k*q for various q
        # But to be safe, let's use all x in [1, max_x] if feasible, else sample
        
        if max_x <= 2000:
            x_vals = np.arange(1, max_x + 1)
        else:
            # Sample x values: include multiples of each k, and some random ones
            x_set = set()
            for k in ks:
                for q in range(1, max_x // k + 1):
                    x_set.add(k * q)
            x_vals = np.array(sorted(x_set))
            if len(x_vals) > 3000:
                # Subsample
                idx = np.linspace(0, len(x_vals) - 1, 3000).astype(int)
                x_vals = x_vals[idx]
        
        n_constraints = len(x_vals)
        
        # Build A_ub: each row is [floor(x/k) for k in ks]
        # Constraint: sum_k f(k) * floor(x/k) <= 1
        A_ub = np.zeros((n_constraints, m))
        for i, x in enumerate(x_vals):
            A_ub[i, :] = np.floor(x / ks)
        
        b_ub = np.ones(n_constraints)
        
        # Bounds: -10 <= f_k <= 10
        bounds = [(-10, 10)] * m
        
        # Solve LP
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                f_vals = res.x
                
                # Compute S = sum_k f(k)/k for k >= 2
                S = np.sum(f_vals / ks)
                f1 = -S
                
                # Build domain and values
                domain = np.concatenate(([1], ks))
                values = np.concatenate(([f1], f_vals))
                
                # Verify M(x) <= 1.0001 for ALL x in [1, 10*K_max]
                K_actual = int(np.max(domain))
                max_x_verify = 10 * K_actual
                
                # Efficient verification
                M = np.zeros(max_x_verify + 1)
                for k in ks:
                    fk = f_vals[ks == k][0] if np.any(ks == k) else 0
                    # floor(x/k) for x = 1..max_x_verify
                    # floor(x/k) = q for x in [q*k, (q+1)*k - 1]
                    for q in range(1, max_x_verify // k + 1):
                        start_x = q * k
                        end_x = min((q + 1) * k - 1, max_x_verify)
                        M[start_x:end_x+1] += fk * q
                
                # Also account for f(1) * floor(x/1) = f1 * x
                M[1:] += f1 * np.arange(1, max_x_verify + 1)
                
                if np.max(M) <= 1.0001:
                    # Feasible! Compute the metric
                    a = -np.sum(values * np.log(domain) / domain)
                    # But we need to use clipped values
                    clipped = np.clip(values, -10, 10)
                    S_clip = np.sum(clipped / domain)
                    clipped[domain == 1] = clipped[domain == 1] - S_clip
                    a = -np.sum(clipped * np.log(domain) / domain)
                    
                    if a > best_score:
                        best_score = a
                        best_domain = domain.copy()
                        best_values = values.copy()
        except Exception:
            pass
    
    # If no LP solution found, fall back to Mobius
    if best_domain is None:
        K = 200000
        mu = _compute_mobius(K)
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[domain].astype(np.float64)
    else:
        domain = best_domain
        values = best_values
    
    return domain, values

"""
Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2.

Strategy:
1. Exploit cyclic symmetry. The objective function is invariant under cyclic shifts.
   Therefore, the global optimum is likely to be periodic or nearly periodic.
   We restrict the search to periodic structures with period p, where p is a divisor of N=100.
   This reduces the search space from 100 dimensions to p dimensions.
   
2. For each candidate period p (1, 2, 4, 5, 10, 20, 25, 50, 100), we define h as a repetition 
   of a base vector u of length p.
   The constraint sum(h) = N/2 implies p * sum(u) = N/2 => sum(u) = N/(2p).
   The bounds 0 <= h <= 1 imply 0 <= u <= 1.
   
3. The objective is max_k c[k] where c = np.correlate(h, 1-h, mode='full').
   This is a function of u. We minimize this max using scipy.optimize.minimize (SLSQP) 
   with multiple random restarts to avoid local minima. Since the dimension p is small 
   (<= 100, and typically much smaller for better structures), SLSQP is very effective.

4. We evaluate all valid divisors p and return the best result found.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import linprog # Not used directly, but available. We use minimize.

N = 100
DEADLINE = 140.0  # Safety margin below 150s kill
_START = time.time()

def remaining_time():
    return DEADLINE - (time.time() - _START)

def objective(h):
    """Compute the raw value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project_to_periodic(u, p):
    """
    Construct the full vector h of length N from base vector u of length p.
    u is repeated N/p times.
    """
    reps = N // p
    h = np.tile(u, reps)
    return h

def solve_periodic(p, rng, time_budget=5.0):
    """
    Solve the problem for a fixed period p.
    Returns the best h found for this period.
    """
    if p > N or N % p != 0:
        return None, float('inf')
    
    # The base vector u has length p.
    # Constraints:
    # 1. 0 <= u_i <= 1
    # 2. sum(u) = N / (2 * p)  (since sum(h) = (N/p) * sum(u) = N/2)
    
    target_sum = N / (2.0 * p)
    
    # Objective function in terms of u
    def obj(u):
        h = project_to_periodic(u, p)
        return objective(h)
    
    # Constraints for SLSQP
    # Equality constraint: sum(u) - target_sum = 0
    cons = ({'type': 'eq', 'fun': lambda u: np.sum(u) - target_sum})
    
    # Bounds: 0 <= u <= 1
    bounds = [(0, 1) for _ in range(p)]
    
    best_h = None
    best_val = float('inf')
    
    # Number of restarts depends on dimension and time budget
    # Smaller p allows more restarts
    num_restarts = max(10, int(1000 / p))
    
    start_time = time.time()
    
    for i in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization satisfying sum constraint approximately
        # We can initialize with uniform noise and project
        u_init = rng.uniform(0.1, 0.9, p)
        # Project to sum constraint
        # Simple projection: shift and clip
        for _ in range(50):
            s = u_init.sum()
            if abs(s - target_sum) < 1e-10:
                break
            u_init += (target_sum - s) / p
            u_init = np.clip(u_init, 0, 1)
        
        # Run SLSQP
        try:
            res = minimize(obj, u_init, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success or res.fun < best_val:
                u_opt = res.x
                # Ensure strict feasibility
                u_opt = np.clip(u_opt, 0, 1)
                # Final projection to sum
                for _ in range(50):
                    s = u_opt.sum()
                    if abs(s - target_sum) < 1e-12:
                        break
                    u_opt += (target_sum - s) / p
                    u_opt = np.clip(u_opt, 0, 1)
                
                h_candidate = project_to_periodic(u_opt, p)
                val = objective(h_candidate)
                
                if val < best_val:
                    best_val = val
                    best_h = h_candidate
                    
        except Exception:
            continue
            
    return best_h, best_val

def construct():
    """
    Search over all possible periods p (divisors of 100) to find the global minimum.
    """
    rng = np.random.default_rng(42)
    
    # Divisors of 100: 1, 2, 4, 5, 10, 20, 25, 50, 100
    divisors = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # We allocate time based on likelihood of finding better solutions.
    # Smaller periods are more constrained but often have better structure.
    # However, the true optimum might not be perfectly periodic.
    # But the "cyclic symmetry" insight suggests we should look for periodic solutions first.
    # If we find a very good periodic solution, we can use it as a starting point for a full-space local search.
    
    # Time allocation: 
    # p=1: 1s
    # p=2: 2s
    # p=4: 3s
    # p=5: 3s
    # p=10: 5s
    # p=20: 5s
    # p=25: 5s
    # p=50: 5s
    # p=100: 10s (full space, but using the periodic structure as init might help? 
    #          Actually, for p=100, we just do SLSQP on 100 vars. It's slow but let's try.)
    
    time_alloc = {
        1: 1.0,
        2: 2.0,
        4: 3.0,
        5: 3.0,
        10: 5.0,
        20: 5.0,
        25: 5.0,
        50: 5.0,
        100: 10.0
    }
    
    # Sort by potential value? Smaller p is easier to solve.
    # Let's iterate through divisors.
    
    for p in divisors:
        if remaining_time() < 2.0:
            break
            
        alloc = time_alloc.get(p, 2.0)
        # Cap allocation to remaining time
        alloc = min(alloc, remaining_time() - 1.0)
        
        if alloc < 0.5:
            break
            
        h_candidate, val_candidate = solve_periodic(p, rng, time_budget=alloc)
        
        if h_candidate is not None and val_candidate < best_val:
            best_val = val_candidate
            best_h = h_candidate.copy()
            
    # Final Refinement: 
    # If we found a good periodic solution, use it as a seed for a quick full-space local search 
    # to break periodicity if it helps.
    if remaining_time() > 5.0 and best_val < 1.0: # Only refine if we have a good candidate
        # Simple local search on the full vector
        h = best_h.copy()
        curr_val = objective(h)
        
        # Use the same local search logic as the parent but seeded with our best periodic solution
        # This is a hybrid: Global structure from periodicity, local refinement in full space.
        
        def local_search_refine(h_init, max_iter=500, step_size=0.05):
            h = h_init.copy()
            current_val = objective(h)
            curr_step = step_size
            rng_local = np.random.default_rng(999)
            
            for _ in range(max_iter):
                if remaining_time() < 2.0:
                    break
                
                improved = False
                for _ in range(20):
                    idx = rng_local.integers(0, N)
                    delta = rng_local.uniform(-curr_step, curr_step)
                    h_new = h.copy()
                    h_new[idx] += delta
                    
                    # Project to feasible set (bounds and sum)
                    h_new = np.clip(h_new, 0, 1)
                    target = N / 2.0
                    for _ in range(20):
                        s = h_new.sum()
                        if abs(s - target) < 1e-10:
                            break
                        h_new += (target - s) / N
                        h_new = np.clip(h_new, 0, 1)
                    
                    if np.allclose(h_new, h, atol=1e-9):
                        continue
                        
                    new_val = objective(h_new)
                    if new_val < current_val:
                        h = h_new
                        current_val = new_val
                        improved = True
                        curr_step *= 0.9
                        break
                if not improved:
                    curr_step *= 0.8
                    if curr_step < 1e-4:
                        break
                        
            return h, current_val

        h_ref, val_ref = local_search_refine(best_h)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
            
    # Final sanity check
    best_h = np.clip(best_h, 0, 1)
    target = N / 2.0
    for _ in range(100):
        s = best_h.sum()
        if abs(s - target) < 1e-12:
            break
        best_h += (target - s) / N
        best_h = np.clip(best_h, 0, 1)
        
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h

# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END

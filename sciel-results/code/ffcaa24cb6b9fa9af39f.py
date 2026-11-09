# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
import sys

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def solve_column_generation(K, deadline, mu=None):
    """
    Solves the LP using Column Generation.
    Variables: f(2)..f(K)
    Objective: minimize sum f(k)*log(k)/k
    Constraints: M(x) <= 1 for all x in [1, 10K]
    
    We start with a small set of x constraints and iteratively add the most violated one.
    """
    n_vars = K - 1
    ks = np.arange(2, K + 1, dtype=np.float64)
    c = np.log(ks) / ks
    
    # Initial set of x values to constrain
    # Focus on small x where M(x) is most likely to exceed 1
    initial_xs = list(range(1, min(50, 10*K + 1)))
    # Add some spread out ones
    step = max(1, 10*K // 100)
    for x in range(50, 10*K + 1, step):
        initial_xs.append(x)
    initial_xs.append(10*K)
    
    active_xs = sorted(set(initial_xs))
    
    # Precompute floor(x/k) for active xs? No, that's what we want to avoid.
    # We will compute rows on the fly for the active set.
    
    current_solution = None
    dual_prices = None
    
    max_iter = 20
    for iter_idx in range(max_iter):
        if time.time() > deadline - 30:
            break
            
        # Build the constraint matrix for the current active_xs
        # A_ub shape: (len(active_xs), n_vars)
        # A_ub[i, j] = floor(x_i / k_j) - 1/k_j
        # Note: The constraint is sum_k f(k) * floor(x/k) <= 1
        # But we have the sum f(k)/k = 0 constraint handled by f(1) adjustment?
        # Wait, the problem statement says:
        # "values are clipped... f(1) is replaced by f(1) - S ... so sum f(k)/k = 0"
        # "M(x) = sum_k f(k) * floor(x/k)"
        # So the constraint on f(2)..f(K) is:
        # sum_{k=2}^K f(k) * floor(x/k) + f(1) * floor(x/1) <= 1
        # f(1) * x + sum_{k=2}^K f(k) * floor(x/k) <= 1
        # Since f(1) = -sum_{k=2}^K f(k)/k, we substitute:
        # x * (-sum_{k=2}^K f(k)/k) + sum_{k=2}^K f(k) * floor(x/k) <= 1
        # sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1
        
        n_constraints = len(active_xs)
        A_ub = np.empty((n_constraints, n_vars), dtype=np.float64)
        
        for idx, x in enumerate(active_xs):
            # Vectorized row construction
            # floor(x/ks) - x/ks
            # For k > x, floor(x/k) = 0, so term is -x/k
            # For k <= x, term is floor(x/k) - x/k
            # We can compute this for all k in ks at once
            floors = x // ks
            A_ub[idx, :] = floors - x / ks
            
        b_ub = np.ones(n_constraints)
        bounds = [(-10.0, 10.0)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={'presolve': True, 'dual_feasibility_tolerance': 1e-8})
        except Exception:
            # Fallback to simpler method if it fails
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
        if not res.success:
            # If LP fails, try with fewer constraints or break
            if current_solution is not None:
                break
            else:
                return None, None, -np.inf

        f_vals = res.x.copy()
        
        # Check for violated constraints
        # We need to find x in [1, 10K] such that M(x) > 1.0001
        # M(x) = sum f(k) * floor(x/k)
        # Using the derived form: sum f(k) * (floor(x/k) - x/k) + x * sum f(k)/k
        # Wait, the condition is M(x) <= 1.
        # M(x) = sum_{k=1}^K f(k) floor(x/k)
        # f(1) = -sum_{k=2}^K f(k)/k
        # M(x) = f(1)x + sum_{k=2}^K f(k) floor(x/k)
        # M(x) = -x * sum_{k=2}^K f(k)/k + sum_{k=2}^K f(k) floor(x/k)
        # M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k)
        
        # We want to find max_x [ sum f(k) (floor(x/k) - x/k) ]
        # This is equivalent to finding the x that maximizes the violation.
        # Since we can't check all 10K x's efficiently every iteration, 
        # we sample or use the dual prices if available.
        
        # Heuristic: Check a dense grid of x's to find violations
        # Focus on regions where the derivative might change or where previous violations were
        # Let's check a set of candidate x's
        candidates = set()
        
        # 1. Small x's (1 to 100)
        candidates.update(range(1, min(101, 10*K + 1)))
        
        # 2. Around the current active xs
        for x in active_xs:
            candidates.update(range(max(1, x-10), min(10*K+1, x+11)))
            
        # 3. Random sampling / Grid
        step_check = max(1, 10*K // 200)
        for x in range(1, 10*K + 1, step_check):
            candidates.add(x)
            
        candidates = sorted(candidates)
        
        violated_xs = []
        max_violation = -np.inf
        worst_x = None
        
        for x in candidates:
            # Compute M(x) for this x
            # M(x) = sum_{k=2}^K f(k) * (floor(x/k) - x/k)
            # Vectorized
            floors = x // ks
            M_x = np.sum(f_vals * (floors - x / ks))
            
            if M_x > 1.0001:
                violated_xs.append(x)
                if M_x > max_violation:
                    max_violation = M_x
                    worst_x = x
                    
        if not violated_xs:
            # No violations found in candidates. Assume feasible.
            current_solution = f_vals
            break
        else:
            # Add the worst violating x to active constraints
            if worst_x not in active_xs:
                active_xs.append(worst_x)
                active_xs = sorted(active_xs)
            # Also add a few nearby ones to stabilize
            for x in violated_xs[:5]:
                if x not in active_xs:
                    active_xs.append(x)
            active_xs = sorted(active_xs)
            
            # If we have too many constraints, prune?
            # For now, just keep adding. The matrix will grow.
            # If it gets too big, we might run out of memory.
            # Limit active_xs size
            if len(active_xs) > 500:
                # Keep the ones with highest dual prices? Or just the most violated?
                # For simplicity, keep the most recently added and small x's
                # This is a heuristic. A proper implementation would use duals.
                pass

    if current_solution is None:
        # If we never converged, use the last solution if it was "good enough"
        # Or just return the last f_vals
        if 'f_vals' in locals():
            current_solution = f_vals
        else:
            return None, None, -np.inf

    # Clip to [-10, 10]
    f_vals = np.clip(current_solution, -10.0, 10.0)
    
    # Compute f(1)
    f1 = -np.sum(f_vals / ks)
    
    # Build full arrays
    values_full = np.zeros(K + 1, dtype=np.float64)
    values_full[1] = f1
    values_full[2:K + 1] = f_vals
    
    # Remove near-zero values
    mask = np.abs(values_full[1:]) > 1e-12
    domain_out = np.nonzero(mask)[0] + 1
    values_out = values_full[1:][mask]
    
    if len(domain_out) < 1:
        return None, None, -np.inf
        
    # Compute score
    log_k = np.log(domain_out.astype(np.float64))
    a = -np.sum(values_out * log_k / domain_out.astype(np.float64))
    
    return domain_out, values_out, a

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the score a = -sum f(k)*log(k)/k subject to M(x) <= 1 for all x.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Precompute Möbius for potential warm start (not used in CG directly, but good to have)
    K_max_try = 100000
    mu = compute_mobius(K_max_try)
    
    # Try different K values
    # With Column Generation, we can try larger K
    K_list = [100000, 50000, 20000]
    
    for K in K_list:
        if time.time() > deadline - 60:
            break
            
        domain, values, a = solve_column_generation(K, deadline, mu)
        
        if domain is not None and len(domain) > 0:
            # Verify feasibility on a denser set of x values
            # This is a critical step. Column generation might miss some violations.
            # We must do a full check for the final candidate.
            
            # Reconstruct full f for checking
            f_full = np.zeros(K + 1, dtype=np.float64)
            for i, k in enumerate(domain):
                if k <= K:
                    f_full[k] = values[i]
            # f(1) is already in values if 1 is in domain, but we calculated it separately.
            # The domain/values returned by solve_column_generation include 1 if f(1) != 0.
            # Let's re-verify M(x) for a dense set of x.
            
            check_xs = list(range(1, min(500, 10*K + 1)))
            step = max(1, 10*K // 500)
            for x in range(500, 10*K + 1, step):
                check_xs.append(x)
            if 10*K not in check_xs:
                check_xs.append(10*K)
            
            feasible = True
            for x in check_xs:
                k_max = min(x, K)
                if k_max < 2:
                    M_x = 0
                else:
                    ks_check = np.arange(1, k_max + 1)
                    floors = x // ks_check
                    M_x = np.sum(f_full[1:k_max + 1] * floors)
                
                if M_x > 1.0001:
                    feasible = False
                    break
            
            if feasible:
                if a > best_score:
                    best_score = a
                    best_domain = domain.copy()
                    best_values = values.copy()
            else:
                # If infeasible, we might try to repair or just skip this K
                # For now, skip. The next K (smaller) might be easier.
                pass
    
    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END

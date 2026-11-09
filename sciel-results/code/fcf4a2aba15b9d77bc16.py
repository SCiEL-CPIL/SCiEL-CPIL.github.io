import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre, roots_genlaguerre

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    m = 12
    n = 1
    
    # Generate candidate root configurations
    candidates = []
    
    # 1. Arithmetic progression with scale 10 (current best: 33.175)
    arith_10 = [4.0 * 10.0 * (i + 1) for i in range(12)]
    candidates.append(('arith_10', arith_10))
    
    # 2. Laguerre zeros of L_12^(1)(x) - the generalized Laguerre with alpha=1
    try:
        lag_roots = roots_genlaguerre(m, 1)
        lag_roots_real = np.sort(lag_roots.real)
        # Scale to match the range of arith_10 (40 to 480)
        lag_scale = 480.0 / lag_roots_real[-1]
        lag_scaled = [float(r) * lag_scale for r in lag_roots_real]
        candidates.append(('lag_scaled', lag_scaled))
        
        # Also try unscaled Laguerre zeros
        lag_unscaled = [float(r) for r in lag_roots_real]
        candidates.append(('lag_unscaled', lag_unscaled))
        
        # Try various scales for Laguerre zeros
        for scale in [5, 10, 20, 50, 100]:
            lag_s = [float(r) * scale for r in lag_roots_real]
            candidates.append((f'lag_scale_{scale}', lag_s))
    except Exception:
        pass
    
    # 3. Geometric progression
    geo_roots = [4.0 * (2.0 ** i) for i in range(12)]
    candidates.append(('geo', geo_roots))
    
    # 4. Square root progression
    sqrt_roots = [4.0 * ((i + 1) ** 0.5) * 10.0 for i in range(12)]
    candidates.append(('sqrt', sqrt_roots))
    
    # 5. Try to refine the arithmetic progression with local optimization
    # Use a smooth surrogate: minimize the condition number of the Vandermonde-like matrix
    # or maximize the spread in a controlled way
    
    # For now, let's try a slightly perturbed arithmetic progression
    for pert in [0.5, 1.0, 2.0, 5.0]:
        pert_roots = [4.0 * 10.0 * (i + 1) + pert * np.sin(i) for i in range(12)]
        if all(r > 0 for r in pert_roots):
            candidates.append((f'pert_{pert}', pert_roots))
    
    # 6. Try arithmetic with different base and step
    for base in [2, 3, 5, 8]:
        for step in [3, 5, 8, 12]:
            roots = [base * (i + 1) * step for i in range(12)]
            if all(r > 0 for r in roots):
                candidates.append((f'base{base}_step{step}', roots))
    
    # Select the best candidate based on a heuristic proxy
    # Since we can't compute the actual C4 bound, use a proxy:
    # The bound likely decreases with more "spread out" roots, but not too much
    # Use the geometric mean of roots as a proxy (larger = better spread)
    
    best_candidate = None
    best_proxy = -np.inf
    
    for name, roots in candidates:
        if len(roots) != 12:
            continue
        if not all(r > 0 for r in roots):
            continue
        # Check if roots are distinct
        if len(set(roots)) != 12:
            continue
        
        # Proxy: geometric mean (log of product / 12)
        log_product = np.sum(np.log(roots)) / 12.0
        # Also consider the spread
        spread = np.max(roots) - np.min(roots)
        
        # Combine: prefer larger geometric mean and moderate spread
        proxy = log_product + 0.1 * np.log(spread + 1)
        
        if proxy > best_proxy:
            best_proxy = proxy
            best_candidate = roots
    
    if best_candidate is None:
        best_candidate = arith_10
    
    # Try local optimization around the best candidate
    try:
        x0 = np.array(best_candidate, dtype=float)
        
        def objective(x):
            """Surrogate objective: minimize a proxy for the C4 bound.
            The C4 bound likely involves ratios of norms of polynomials.
            For double roots, the polynomial is P(x) = prod((x - r_i)^2).
            The bound might involve ||P||_2 / ||P'||_2 or similar.
            Let's use the ratio of the L2 norm to the max value as a proxy.
            """
            if np.any(x <= 0) or np.any(np.diff(np.sort(x)) <= 0):
                return 1e10
            
            # Compute a simple proxy: the condition number of the Vandermonde matrix
            # or the ratio of product of roots to sum of roots
            sorted_x = np.sort(x)
            
            # Use the geometric mean as a proxy for "how spread out" the roots are
            # Larger geometric mean = roots are further from origin = better
            log_prod = np.sum(np.log(sorted_x))
            
            # Also penalize if roots are too close together
            min_gap = np.min(np.diff(sorted_x))
            if min_gap < 1e-6:
                return 1e10
            
            # The proxy: maximize log_prod (equivalent to minimizing -log_prod)
            # But we also want roots not to be too large (numerical stability)
            return -log_prod + 0.01 * np.max(sorted_x)
        
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.success and np.all(result.x > 0):
            optimized_roots = [float(r) for r in result.x]
            # Verify they're distinct and positive
            if len(set(optimized_roots)) == 12 and all(r > 0 for r in optimized_roots):
                # Use the optimized roots
                best_candidate = optimized_roots
    except Exception:
        pass
    
    # Final check: ensure we have 12 positive distinct roots
    final_roots = [float(r) for r in best_candidate]
    if len(final_roots) != 12 or not all(r > 0 for r in final_roots):
        final_roots = arith_10
    
    if time.time() - t0 > deadline:
        return arith_10
    
    return final_roots

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre, roots_genlaguerre
from scipy.optimize import minimize, differential_evolution, basinhopping
import time

# Precompute quadrature nodes for high precision
# 200 points is usually sufficient for 12 double roots (degree 24)
_NQ = 200
_xq, _wq = np.polynomial.laguerre.laggauss(_NQ)

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for given double-root positions.
    C4 = int(P^2 e^-x) / (int(P e^-x))^2
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    # Vectorized calculation of P(x) = prod (x - r_i)^2
    # diff shape: (NQ, 12)
    diff = _xq[:, np.newaxis] - roots[np.newaxis, :]
    
    # Calculate log(P) for stability, then exp
    # P = prod (x-r)^2 = (prod (x-r))^2
    # log(P) = 2 * sum(log(|x-r|))
    # We need P and P^2.
    # P = exp(2 * sum(log|diff|))
    # P^2 = exp(4 * sum(log|diff|))
    
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = np.exp(2.0 * log_P) # Or P*P, but exp(2*log_P) is same and maybe safer if P is huge? 
                             # Actually P*P is fine if P is not inf.
    
    # Integrals
    int_P = np.sum(_wq * P)
    int_P2 = np.sum(_wq * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
        
    c4 = int_P2 / (int_P * int_P)
    return c4

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin
    
    # 1. Generate diverse initial points
    initial_points = []
    
    # Standard Laguerre roots (alpha=0)
    x0, _ = roots_laguerre(12)
    initial_points.append(x0.copy())
    
    # Generalized Laguerre roots for various alpha
    # roots_genlaguerre(n, alpha) returns roots of L_n^(alpha)(x)
    # alpha > -1
    for alpha in [-0.5, 0.5, 1.0, 1.5, 2.0, 3.0]:
        try:
            x_alpha, _ = roots_genlaguerre(12, alpha)
            x_alpha = np.sort(x_alpha)
            if np.all(x_alpha > 0):
                initial_points.append(x_alpha.copy())
        except Exception:
            pass
            
    # Log-uniform random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Log uniform between 0.1 and 100
        r = np.exp(rng.uniform(np.log(0.1), np.log(100), size=12))
        r = np.sort(r)
        initial_points.append(r.copy())
        
    # Power law starts: r_i = c * i^k
    for k in [0.5, 1.0, 1.5, 2.0]:
        for c in [0.1, 0.5, 1.0, 2.0]:
            r = c * np.arange(1, 13) ** k
            r = np.sort(r)
            if np.all(r > 0) and r[-1] < 200: # Keep within reasonable bounds
                initial_points.append(r.copy())

    # 2. Local optimization from each start
    best_roots = None
    best_score = np.inf
    bounds = [(1e-6, None)] * 12
    
    for x_init in initial_points:
        if time.time() > deadline - 60:
            break
            
        try:
            # L-BFGS-B
            res = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_roots = res.x.copy()
        except Exception:
            pass

    if best_roots is None:
        # Fallback to standard Laguerre
        best_roots = initial_points[0].copy()
        best_score = _compute_c4_bound(best_roots)

    # 3. Basin Hopping to escape local minima
    # Use the best root found so far as the starting point
    # Basin Hopping uses a step size. We need to estimate a good step size.
    # Average spacing of roots is a good heuristic.
    sorted_best = np.sort(best_roots)
    spacings = np.diff(sorted_best)
    avg_spacing = np.median(spacings) if len(spacings) > 0 else 1.0
    
    # Define a step function for basin hopping
    def step(roots):
        # Random perturbation
        noise = np.random.normal(0, avg_spacing * 0.5, size=len(roots))
        new_roots = roots + noise
        # Ensure positivity
        new_roots = np.maximum(new_roots, 1e-6)
        return new_roots

    # Minimizer for basin hopping
    def minimizer(roots):
        res = minimize(
            _objective,
            roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-15}
        )
        return {'xf': res.x, 'fval': res.fun}

    # Run Basin Hopping
    if time.time() < deadline - 120:
        try:
            bh = basinhopping(
                _objective,
                x0=best_roots,
                minimizer=minimizer,
                stepsize=avg_spacing * 1.0,
                niter=20, # Number of basin hopping steps
                T=1.0,
                seed=42,
                accept_test=None,
                callback=None
            )
            
            if bh.fun < best_score:
                best_score = bh.fun
                best_roots = bh.x.copy()
        except Exception:
            pass

    # 4. Differential Evolution for global search (if time remains)
    # Use tighter bounds based on the best solution found
    if time.time() < deadline - 30:
        # Determine bounds from best solution
        # Expand range by factor 2
        min_r = min(1e-3, np.min(best_roots) * 0.1)
        max_r = max(10.0, np.max(best_roots) * 2.0)
        
        de_bounds = [(min_r, max_r)] * 12
        
        try:
            de_res = differential_evolution(
                _objective,
                bounds=de_bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=True, # Polish with L-BFGS-B
                updating='immediate',
                workers=1
            )
            
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_roots = de_res.x.copy()
        except Exception:
            pass

    # 5. Final high-precision polish
    if time.time() < deadline - 5:
        try:
            final_res = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if final_res.fun < best_score:
                best_score = final_res.fun
                best_roots = final_res.x.copy()
        except Exception:
            pass

    # Ensure sorted and valid
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0):
        # Fallback
        x0, _ = roots_laguerre(12)
        best_roots = x0
        
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END

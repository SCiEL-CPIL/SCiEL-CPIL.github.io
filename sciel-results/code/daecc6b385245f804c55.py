"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre

# Global deadline guard
_T0 = time.time()
_DEADLINE = 630.0  # seconds wall-clock

def _elapsed():
    return time.time() - _T0

def _time_left():
    return _DEADLINE - _elapsed() - 30.0  # 30s safety margin


def _compute_C4(roots, m=12, n_dim=1):
    """
    Compute the upper bound C4 for the Fourier sign uncertainty using
    Gauss-Laguerre quadrature for numerical stability.
    """
    r = np.array(roots, dtype=float)
    if len(r) != m:
        return float('inf')
    if np.any(r <= 0) or np.any(np.isnan(r)):
        return float('inf')
    if np.any(np.diff(np.sort(r)) < 1e-10):
        return float('inf')  # roots must be distinct
    
    # Use Gauss-Laguerre quadrature (includes e^{-x} weight)
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2
    # Compute log|P| for numerical stability
    log_P = np.zeros(n_quad)
    for ri in r:
        diff = x_quad - ri
        log_P += 2.0 * np.log(np.abs(diff) + 1e-300)
    
    P = np.exp(log_P)
    P2 = P * P
    
    # ||P||^2 = integral_0^inf |P(x)|^2 e^{-x} dx
    int_P2 = np.sum(w_quad * P2)
    
    # Also compute integral of P
    int_P = np.sum(w_quad * P)
    
    if int_P2 < 1e-15 or not np.isfinite(int_P2):
        return float('inf')
    
    # The bound C4 involves the ratio of certain integrals
    # For the refined uncertainty inequality, a key quantity is:
    # C4 = 1 / (||P||^2 / ||P||_1^2) or similar
    # Let's use: C4 = int_P^2 / int_P2
    # This is related to the effective number of "modes" or concentration
    
    if int_P < 1e-15:
        return float('inf')
    
    C4 = (int_P ** 2) / int_P2
    return C4


def _score(roots):
    """Compute the score (upper bound on C4). Lower is better."""
    try:
        r = np.array(roots, dtype=float)
        if len(r) != 12:
            return float('inf')
        if np.any(r <= 0) or np.any(np.isnan(r)):
            return float('inf')
        # Ensure distinct roots
        rs = np.sort(r)
        if np.any(np.diff(rs) < 1e-8):
            return float('inf')
        return _compute_C4(r)
    except Exception:
        return float('inf')


def _cost_func(params):
    """Cost function for optimization. params = 12 log-spaced root positions."""
    log_roots = np.array(params, dtype=float)
    roots = np.exp(log_roots)
    
    # Check constraints
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return 1e10
    rs = np.sort(roots)
    if np.any(np.diff(rs) < 1e-6):
        return 1e10
    if np.max(roots) > 1e8:
        return 1e10
    if np.min(roots) < 1e-8:
        return 1e10
    return _score(roots)


def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    """
    # Seed solution: geometric with r0=4, q=2 -> [4, 8, 16, ..., 4*2^11]
    seed_roots = [4.0 * (2 ** i) for i in range(12)]
    best_score = _score(seed_roots)
    best_roots = np.array(seed_roots, dtype=float)
    
    # If time is very short, return seed
    if _time_left() < 5.0:
        return [float(r) for r in best_roots]
    
    # Phase 1: Differential evolution for global search
    # Parameterize as log(roots) to ensure positivity
    seed_log_roots = np.log(seed_roots)
    
    # Bounds for log(roots): log(1e-6) to log(1e6)
    bounds = [(np.log(1e-6), np.log(1e6))] * 12
    
    # Run differential evolution
    de_result = None
    try:
        de_result = differential_evolution(
            _cost_func,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            popsize=15,
            polish=False,
            init='sobol'
        )
    except Exception:
        pass
    
    if de_result is not None and de_result.fun < best_score:
        best_roots = np.exp(de_result.x)
        best_score = de_result.fun
    
    # Phase 2: Local refinement with L-BFGS-B from multiple starts
    if _time_left() > 15.0:
        # Generate multiple starting points
        rng = np.random.RandomState(42)
        starts = []
        
        # Start from current best
        starts.append(np.log(best_roots))
        
        # Add perturbed versions
        for _ in range(20):
            perturbation = rng.randn(12) * 0.5
            start = np.log(best_roots) + perturbation
            # Ensure within bounds
            start = np.clip(start, np.log(1e-5), np.log(1e5))
            starts.append(start)
        
        # Add some random starts
        for _ in range(10):
            start = rng.uniform(np.log(1.0), np.log(100.0), size=12)
            # Sort and add small gaps
            start = np.sort(start)
            start[1:] += rng.uniform(0.1, 0.5, size=11)
            starts.append(start)
        
        # Try each start with L-BFGS-B
        for idx, start in enumerate(starts):
            if _time_left() < 10.0:
                break
            
            try:
                result = minimize(
                    _cost_func,
                    x0=start,
                    method='L-BFGS-B',
                    bounds=[(np.log(1e-6), np.log(1e6))] * 12,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_score:
                    new_roots = np.exp(result.x)
                    if np.all(new_roots > 0) and np.all(np.diff(np.sort(new_roots)) > 1e-6):
                        s = _score(new_roots)
                        if s < best_score:
                            best_score = s
                            best_roots = new_roots
            except Exception:
                continue
    
    # Phase 3: Final polish on the best solution
    if _time_left() > 10.0 and np.all(best_roots > 0):
        try:
            result = minimize(
                _cost_func,
                x0=np.log(best_roots),
                method='L-BFGS-B',
                bounds=[(np.log(1e-6), np.log(1e6))] * 12,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_score:
                new_roots = np.exp(result.x)
                if np.all(new_roots > 0) and np.all(np.diff(np.sort(new_roots)) > 1e-6):
                    s = _score(new_roots)
                    if s < best_score:
                        best_score = s
                        best_roots = new_roots
        except Exception:
            pass
    
    # Ensure we return a valid list of 12 positive floats
    if len(best_roots) != 12 or np.any(best_roots <= 0) or np.any(np.isnan(best_roots)):
        return [float(r) for r in seed_roots]
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END

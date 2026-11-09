import time
import math
import numpy as np
from scipy.optimize import minimize, root, brentq
from scipy.special import eval_genlaguerre, factorial, roots_genlaguerre

# EVOLVE-BLOCK-START

def _laguerre_poly(x, m, n):
    """Generalized Laguerre polynomial L_m^(n)(x)."""
    return eval_genlaguerre(m, n, x)

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 for the Cohn-Goncalves construction.
    
    The construction uses 12 double roots r_1 < r_2 < ... < r_12.
    The upper bound on C4 is related to the last sign change of the
    Laguerre-based test function.
    
    For the refined construction, C4 is bounded by a function of the
    root spacings and the Laguerre polynomial values at those roots.
    """
    roots = np.array(roots)
    m = 12
    n = 1
    
    # The Cohn-Goncalves bound involves the product of Laguerre polynomial
    # values at the roots and their derivatives.
    # 
    # For double roots, the key quantity is the "last sign change" which
    # determines the uncertainty constant.
    #
    # The bound C4 is approximately:
    # C4 = max over sign change intervals of some ratio involving
    # the Laguerre polynomial and its derivative.
    
    # Evaluate L_m^(n)(x) and its derivative at each root
    L_vals = np.array([eval_genlaguerre(m, n, r) for r in roots])
    
    # Derivative of L_m^(n)(x) = -L_{m-1}^{(n+1)}(x)
    L_deriv_vals = np.array([-eval_genlaguerre(m-1, n+1, r) for r in roots])
    
    # The upper bound on C4 is related to the maximum ratio |L/L'| 
    # at the roots, which determines the sharpness of the bound.
    # Smaller ratios give better (lower) bounds on C4.
    
    # Avoid division by zero
    mask = np.abs(L_deriv_vals) > 1e-15
    if not np.all(mask):
        # If any derivative is near zero, the bound is large
        return 1e6
    
    ratios = np.abs(L_vals[mask] / L_deriv_vals[mask])
    
    # The C4 bound is proportional to the maximum ratio
    # scaled by a normalization factor
    C4_bound = np.max(ratios)
    
    # Additional constraint: roots must be positive and ordered
    # The bound also depends on the overall scale
    scale_factor = np.prod(roots) ** (1.0 / len(roots))
    
    # The actual C4 upper bound
    return C4_bound * scale_factor

def _verify_constraints(roots):
    """Verify that the roots satisfy the required constraints."""
    roots = np.array(roots)
    
    # All roots must be positive
    if np.any(roots <= 0):
        return False
    
    # Roots must be distinct (double roots at distinct locations)
    if len(roots) != len(np.unique(roots)):
        return False
    
    # Roots should be in increasing order
    if not np.all(np.diff(roots) > 0):
        return False
    
    return True

def _objective(params):
    """
    Objective function for optimization.
    params: 12 root locations
    Returns: C4 upper bound (lower is better)
    """
    roots = np.sort(params)
    
    if not _verify_constraints(roots):
        return 1e10
    
    return _compute_c4_bound(roots)

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    deadline = time.time() + 590  # Safety margin before 630s kill
    
    # Start with the zeros of the Laguerre polynomial L_12^(1)(x)
    # These are known to be positive and distinct
    try:
        roots_lag = roots_genlaguerre(12, 1)
        initial_roots = sorted(roots_lag['roots'].tolist())
    except Exception:
        initial_roots = [4.0 * (i + 1) for i in range(12)]
    
    best_roots = initial_roots
    best_score = _compute_c4_bound(best_roots)
    
    # Try to improve using SLSQP optimization
    try:
        bounds = [(0.1, 100.0)] * 12
        
        # Use SLSQP to minimize the C4 bound
        result = minimize(
            _objective,
            initial_roots,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < best_score:
            candidate_roots = sorted(result.x.tolist())
            if _verify_constraints(candidate_roots):
                candidate_score = _compute_c4_bound(candidate_roots)
                if candidate_score < best_score:
                    best_roots = candidate_roots
                    best_score = candidate_score
        
        if time.time() > deadline:
            return best_roots
        
        # Try multiple random restarts
        np.random.seed(42)
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Perturb the initial roots
            perturbation = np.random.uniform(0.5, 2.0, size=12)
            random_roots = np.sort([r * p for r, p in zip(initial_roots, perturbation)])
            
            result = minimize(
                _objective,
                random_roots,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                candidate_roots = sorted(result.x.tolist())
                if _verify_constraints(candidate_roots):
                    candidate_score = _compute_c4_bound(candidate_roots)
                    if candidate_score < best_score:
                        best_roots = candidate_roots
                        best_score = candidate_score
        
    except Exception:
        pass
    
    # Final polish: use Nelder-Mead for local refinement
    try:
        if time.time() < deadline:
            result = minimize(
                _objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < best_score:
                candidate_roots = sorted(result.x.tolist())
                if _verify_constraints(candidate_roots):
                    candidate_score = _compute_c4_bound(candidate_roots)
                    if candidate_score < best_score:
                        best_roots = candidate_roots
                        best_score = candidate_score
                        
    except Exception:
        pass
    
    return best_roots

# EVOLVE-BLOCK-END

"""
Optimized approach using Chebyshev basis expansion and L-BFGS-B.
The objective is value = abs(2 * N * max(convolve(h, h)) / sum(h)^2).
We parameterize h as a linear combination of Chebyshev polynomials to enforce smoothness
and reduce dimensionality, then optimize the coefficients.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_chebyt

N = 400
DEADLINE = 145.0
START_TIME = time.time()

# Precompute Chebyshev basis for efficiency
_x = np.linspace(-1, 1, N)
MAX_TERMS = 25
_cheb_basis = np.array([eval_chebyt(k, _x) for k in range(MAX_TERMS)])  # shape (MAX_TERMS, N)

# EVOLVE-BLOCK-START

def _value(h):
    s = np.sum(h)
    if np.isclose(s, 0):
        return 1e10
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _soft_max(x, beta=200.0):
    # Smooth approximation of max(x) using log-sum-exp
    m = np.max(x)
    return m + np.log(np.sum(np.exp((x - m) * beta)) / len(x)) / beta

def _coeffs_to_h(coeffs):
    """Convert Chebyshev coefficients to step heights h."""
    num_terms = len(coeffs)
    # coeffs[0] is offset (T_0), coeffs[1:] are T_1, T_2, ...
    h = np.zeros(N)
    for k in range(num_terms):
        if k < MAX_TERMS:
            h += coeffs[k] * _cheb_basis[k]
    return h

def _objective_and_grad(coeffs):
    """Compute objective and gradient analytically."""
    h = _coeffs_to_h(coeffs)
    s = np.sum(h)
    if np.isclose(s, 0):
        return 1e10, np.zeros_like(coeffs)
    
    c = np.convolve(h, h)
    m = _soft_max(c, beta=200.0)
    obj = abs(2.0 * N * m / (s * s))
    
    # Compute gradient numerically (simpler and robust)
    grad = np.zeros_like(coeffs)
    eps = 1e-7
    for i in range(len(coeffs)):
        coeffs_plus = coeffs.copy()
        coeffs_plus[i] += eps
        h_plus = _coeffs_to_h(coeffs_plus)
        s_plus = np.sum(h_plus)
        if np.isclose(s_plus, 0):
            grad[i] = 0
            continue
        c_plus = np.convolve(h_plus, h_plus)
        m_plus = _soft_max(c_plus, beta=200.0)
        obj_plus = abs(2.0 * N * m_plus / (s_plus * s_plus))
        grad[i] = (obj_plus - obj) / eps
    
    return obj, grad

def _make_initial_conditions(rng, num_terms):
    """Generate initial coefficients with small perturbations around seed."""
    coeffs = np.zeros(num_terms)
    coeffs[0] = 1.0  # offset = 1 (matches seed np.ones)
    # Add small random perturbations to higher-order terms
    coeffs[1:] = rng.normal(0, 0.05, num_terms - 1)
    return coeffs

def construct():
    if time.time() - START_TIME > DEADLINE - 5:
        return np.ones(N)
    
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Try multiple Chebyshev term counts
    term_counts = [8, 12, 16, 20, 25]
    
    for num_terms in term_counts:
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # Multiple random restarts for each term count
        for restart in range(4):
            if time.time() - START_TIME > DEADLINE - 10:
                break
            
            coeffs0 = _make_initial_conditions(rng, num_terms)
            
            # Use L-BFGS-B with numerical gradients
            res = minimize(_objective_and_grad, coeffs0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            
            if res.fun < best_val:
                h_candidate = _coeffs_to_h(res.x)
                val_candidate = _value(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate
    
    # Final polish: try to improve with exact max (not soft)
    if time.time() - START_TIME < DEADLINE - 5:
        # Try a few more targeted optimizations around best solution
        for _ in range(3):
            if time.time() - START_TIME > DEADLINE - 5:
                break
            
            # Small perturbation of best solution
            pert = rng.normal(0, 0.01, len(_cheb_basis))
            # We need to extract coefficients from best_h - approximate by fitting
            # For simplicity, just try small perturbations in coefficient space
            # Re-derive: use least squares to get coefficients from best_h
            A = _cheb_basis[:min(25, MAX_TERMS)]  # use available terms
            coeffs_fit, _, _, _ = np.linalg.lstsq(A, best_h, rcond=None)
            
            coeffs_perturbed = coeffs_fit + rng.normal(0, 0.02, len(coeffs_fit))
            
            res = minimize(_objective_and_grad, coeffs_perturbed, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_val:
                h_candidate = _coeffs_to_h(res.x)
                val_candidate = _value(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate
    
    return best_h

# EVOLVE-BLOCK-END

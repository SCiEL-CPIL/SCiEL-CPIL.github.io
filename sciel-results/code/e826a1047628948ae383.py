"""Low-dimensional basis optimization for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_chebyt

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute Chebyshev basis up to degree D
D = 40
_x = np.linspace(-1, 1, N)
_basis = np.zeros((D + 1, N))
for k in range(D + 1):
    _basis[k] = eval_chebyt(k, _x)

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _h_from_coeffs(c):
    """Reconstruct h from Chebyshev coefficients."""
    return _basis.T @ c

def _obj_from_coeffs(c):
    """Objective function in coefficient space."""
    h = _h_from_coeffs(c)
    return _eval_value(h)

def _grad_from_coeffs(c):
    """Gradient of the objective in coefficient space using finite differences or analytic?
    Analytic is better. Let's use numerical for simplicity in low dim, or analytic if easy.
    For D=40, numerical gradient is cheap enough (41 evals).
    But L-BFGS-B needs it. Let's use a central difference approximation.
    """
    eps = 1e-8
    grad = np.zeros_like(c)
    for i in range(len(c)):
        c_plus = c.copy()
        c_minus = c.copy()
        c_plus[i] += eps
        c_minus[i] -= eps
        f_plus = _obj_from_coeffs(c_plus)
        f_minus = _obj_from_coeffs(c_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0

    # Initial best: seed program
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    # Define bounds for coefficients
    # Chebyshev polynomials are bounded by 1 on [-1,1].
    # If we keep coefficients small, h stays bounded.
    # Let's try bounds [-10, 10] for each coefficient.
    bounds = [(-10.0, 10.0)] * (D + 1)
    
    # Strategy 1: Differential Evolution on low-dim space
    # DE is global, good for finding a decent basin.
    if elapsed() < DEADLINE - 30:
        try:
            # Use a subset of bounds or tighter bounds?
            # Let's start with a moderate population size
            res_de = differential_evolution(
                _obj_from_coeffs,
                bounds,
                maxiter=50,
                popsize=20,
                tol=1e-5,
                seed=42,
                polish=False, # We will polish manually
                update_interval=5,
                callback=lambda x, convergence: None # No callback needed
            )
            c_best = res_de.x
            h_candidate = _h_from_coeffs(c_best)
            v_candidate = _eval_value(h_candidate)
            
            if v_candidate < best_val and np.all(np.isfinite(h_candidate)):
                best_val = v_candidate
                best_h = h_candidate.copy()
        except Exception as e:
            pass

    # Strategy 2: Local Refinement with L-BFGS-B from multiple starts
    # Generate diverse starting points in coefficient space
    rng = np.random.RandomState(123)
    starts = []
    
    # 1. From DE result
    if 'c_best' in locals():
        starts.append(c_best)
    
    # 2. Random starts
    for _ in range(20):
        c_rand = rng.uniform(-5, 5, D + 1)
        starts.append(c_rand)
        
    # 3. Structured starts: single frequencies
    for k in [1, 2, 3, 5, 8, 13]:
        c_struct = np.zeros(D + 1)
        c_struct[k] = 1.0
        starts.append(c_struct)
        c_struct_neg = np.zeros(D + 1)
        c_struct_neg[k] = -1.0
        starts.append(c_struct_neg)
        
    # 4. Linear ramp
    c_linear = np.zeros(D + 1)
    # T1 is x, so just setting T1 coeff gives linear h
    c_linear[1] = 1.0
    starts.append(c_linear)
    
    # 5. Quadratic
    c_quad = np.zeros(D + 1)
    c_quad[2] = 1.0
    starts.append(c_quad)

    # Refine each start
    for c_start in starts:
        if elapsed() > DEADLINE - 5:
            break
            
        try:
            # Ensure start is valid
            if not np.all(np.isfinite(c_start)):
                continue
                
            res_min = minimize(
                _obj_from_coeffs,
                c_start,
                jac=_grad_from_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            c_opt = res_min.x
            h_candidate = _h_from_coeffs(c_opt)
            v_candidate = _eval_value(h_candidate)
            
            if v_candidate < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                best_val = v_candidate
                best_h = h_candidate.copy()
                
        except Exception:
            pass

    # Final sanity check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
        best_val = 2.0

    return best_h

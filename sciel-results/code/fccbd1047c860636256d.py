import numpy as np
import time
from scipy.optimize import minimize

N = 50
D = 12  # Number of DCT coefficients to optimize

def _eval(h):
    """
    Computes the objective value for a given height vector h.
    h is assumed to be non-negative. Scale invariance allows us to work with sum(h)=1.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    if np.sum(h) == 0:
        return 0.0
        
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    S = np.sum(terms)
    l2 = (w / 3.0) * S
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def _dct_to_h(coeffs):
    """
    Convert DCT coefficients to non-negative heights.
    coeffs: array of shape (D,)
    Returns: array of shape (N,)
    """
    # Extend coeffs to length N by padding with zeros
    full_coeffs = np.zeros(N)
    full_coeffs[:D] = coeffs
    
    # Inverse DCT (Type II)
    # scipy.fft.idct by default uses Type II? 
    # Let's check scipy.fft.dct/idct types.
    # scipy.fft.dct(x, type=2) is the standard DCT-II.
    # scipy.fft.idct(x, type=2) is the inverse DCT-II.
    # We want to generate h from low-freq coeffs.
    # So we treat coeffs as the DCT-II of h.
    # Thus h = idct(coeffs, type=2).
    # But we need to handle normalization and orthogonality correctly.
    
    # Alternatively, just use np.fft or a manual DCT basis.
    # Let's use a simple cosine basis expansion to avoid scipy.fft normalization pitfalls.
    # h[n] = sum_{k=0}^{D-1} c_k * phi_k[n]
    # where phi_k are cosine basis functions.
    
    n = np.arange(N)
    h = np.zeros(N)
    
    # DC component
    h += coeffs[0] * np.ones(N)
    
    # AC components
    for k in range(1, D):
        # Standard DCT-II basis: phi_k[n] = cos(pi * k * (n + 0.5) / N)
        # Normalization doesn't matter for the shape, but let's keep it standard.
        phi_k = np.cos(np.pi * k * (n + 0.5) / N)
        h += coeffs[k] * phi_k
        
    # Enforce non-negativity
    h = np.abs(h)
    
    # Normalize to sum=1 for optimization stability (scale invariance)
    s = np.sum(h)
    if s == 0:
        return np.ones(N) / N
    h = h / s
    
    return h

def _objective(coeffs):
    """
    Objective function for optimization.
    Returns negative value (for minimization).
    """
    h = _dct_to_h(coeffs)
    val = _eval(h)
    return -val

def _finite_diff_gradient(coeffs, eps=1e-6):
    """
    Compute finite difference gradient of the objective.
    """
    grad = np.zeros_like(coeffs)
    for i in range(D):
        coeffs_plus = coeffs.copy()
        coeffs_plus[i] += eps
        coeffs_minus = coeffs.copy()
        coeffs_minus[i] -= eps
        
        f_plus = _objective(coeffs_plus)
        f_minus = _objective(coeffs_minus)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
        
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Generate diverse initial candidates in DCT coefficient space
    candidates = []
    
    # 1. Uniform (all coeffs zero except DC)
    c_uniform = np.zeros(D)
    c_uniform[0] = 1.0
    candidates.append(c_uniform)
    
    # 2. Gaussian-like (smooth decay in DCT)
    for width in [1, 2, 5]:
        k = np.arange(D)
        c = np.exp(-k**2 / (2 * width**2))
        candidates.append(c)
        candidates.append(-c)
        
    # 3. Boxcar-like (sharp cutoff in DCT)
    for cutoff in [2, 5, 10]:
        c = np.zeros(D)
        c[:cutoff] = 1.0
        candidates.append(c)
        
    # 4. Random
    np.random.seed(42)
    for _ in range(10):
        c = np.random.randn(D)
        candidates.append(c)
        
    best_coeffs = None
    best_val = -np.inf
    
    # Evaluate all candidates
    for c in candidates:
        h = _dct_to_h(c)
        val = _eval(h)
        if val > best_val:
            best_val = val
            best_coeffs = c.copy()
            
    # Polish the best candidate using L-BFGS-B with finite difference gradient
    if time.time() < deadline - 20:
        try:
            res = minimize(
                _objective, 
                best_coeffs, 
                jac=_finite_diff_gradient,
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.fun < -best_val:
                best_val = -res.fun
                best_coeffs = res.x.copy()
                
        except Exception:
            pass
            
    # If we have time left, try a few other local optima from different starts
    if time.time() < deadline - 30:
        # Sort candidates by value
        scored_candidates = []
        for c in candidates:
            h = _dct_to_h(c)
            val = _eval(h)
            scored_candidates.append((val, c))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        for val, c in scored_candidates[:5]:
            if time.time() > deadline - 10:
                break
            if val <= best_val:
                continue # Skip if not better, but try a few distinct ones
                
            try:
                res = minimize(
                    _objective, 
                    c, 
                    jac=_finite_diff_gradient,
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.fun < -best_val:
                    best_val = -res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass

    # Final conversion to heights
    if best_coeffs is None:
        best_coeffs = np.zeros(D)
        best_coeffs[0] = 1.0
        
    best_h = _dct_to_h(best_coeffs)
    
    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from sympy import symbols, hermite_poly, Poly

# Precompute Hermite polynomials symbolically
_x = symbols('x')
H0 = hermite_poly(0, _x)
H4 = hermite_poly(4, _x)
H8 = hermite_poly(8, _x)
H12 = hermite_poly(12, _x)

# Evaluate at 0 for c3 computation
H0_0 = float(H0.subs(_x, 0))
H4_0 = float(H4.subs(_x, 0))
H8_0 = float(H8.subs(_x, 0))
H12_0 = float(H12.subs(_x, 0))

# Convert to polynomial coefficients for fast numerical evaluation
def hermite_coeffs(n):
    """Get coefficients of H_n(x) from highest degree to constant."""
    p = hermite_poly(n, _x)
    return np.array(Poly(p, _x).all_coeffs(), dtype=float)

c0_coeffs = hermite_coeffs(0)
c4_coeffs = hermite_coeffs(4)
c8_coeffs = hermite_coeffs(8)
c12_coeffs = hermite_coeffs(12)

# Pad coefficients to same length (degree 12)
deg_max = 12
def pad_coeffs(c):
    if len(c) == deg_max + 1:
        return c
    pad = np.zeros(deg_max + 1 - len(c))
    return np.concatenate([pad, c])

c0_coeffs_p = pad_coeffs(c0_coeffs)
c4_coeffs_p = pad_coeffs(c4_coeffs)
c8_coeffs_p = pad_coeffs(c8_coeffs)
c12_coeffs_p = pad_coeffs(c12_coeffs)

def build_q_poly(c0, c1, c2):
    """Build q(x) = g(x)/x^2 as a numpy polynomial (highest degree first)."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    g_coeffs = c0 * c0_coeffs_p + c1 * c4_coeffs_p + c2 * c8_coeffs_p + c3 * c12_coeffs_p
    
    # g(x)/x^2: remove last 2 coefficients (constant and linear terms)
    q_coeffs = g_coeffs[:-2]
    
    return q_coeffs

def compute_value(c0, c1, c2):
    """Compute the task value: r^2 / (2*pi) where r is the largest real root
    of q(x)=g(x)/x^2 at which q changes sign."""
    q_coeffs = build_q_poly(c0, c1, c2)
    
    if len(q_coeffs) == 0:
        return float('inf')
        
    # Normalize for numerical stability
    max_coeff = np.max(np.abs(q_coeffs))
    if max_coeff > 1e-10:
        q_coeffs_norm = q_coeffs / max_coeff
    else:
        q_coeffs_norm = q_coeffs
        
    # Find real roots of q
    roots = np.roots(q_coeffs_norm)
    
    # Filter real roots
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    
    if not real_roots:
        return float('inf')
        
    # Sort real roots
    real_roots.sort()
    
    valid_roots = []
    for r in real_roots:
        # Check sign change
        eps = 1e-6 * max(1.0, abs(r))
        q_left = np.polyval(q_coeffs_norm, r - eps)
        q_right = np.polyval(q_coeffs_norm, r + eps)
        
        if q_left * q_right < 0:
            valid_roots.append(r)
            
    if not valid_roots:
        return float('inf')
        
    r = max(valid_roots)
    return r**2 / (2 * np.pi)

def objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    return compute_value(c0, c1, c2)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = compute_value(*seed)
    best_params = seed.copy()
    
    # Define bounds for Differential Evolution
    bounds = [(-5.0, 0.0), (-0.5, 0.5), (-0.05, 0.05)]
    
    # Run Differential Evolution
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            atol=1e-6,
            seed=42,
            init='latinhypercube'
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception as e:
        pass
        
    # Refine with Nelder-Mead if we have time
    if time.time() < deadline - 20:
        try:
            from scipy.optimize import minimize
            res2 = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
            
    return np.array(best_params)
# EVOLVE-BLOCK-END

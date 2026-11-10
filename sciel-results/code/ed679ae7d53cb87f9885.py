"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre, genlaguerre

# EVOLVE-BLOCK-START

_L_coeffs_cache = None

def _get_laguerre_coeffs():
    global _L_coeffs_cache
    if _L_coeffs_cache is None:
        _L_coeffs_cache = []
        alpha = -0.5
        max_deg = 50
        for k in range(26):
            n = 2*k
            L_n = genlaguerre(n, alpha)
            c = np.asarray(L_n.c, dtype=float)
            if len(c) < max_deg + 1:
                c = np.pad(c, (0, max_deg + 1 - len(c)), mode='constant')
            _L_coeffs_cache.append(c)
    return _L_coeffs_cache

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix
    from sympy.polys.orthopolys import genlaguerre as sympy_genlaguerre
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(sympy_genlaguerre(d, alpha, x))
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    L0_vals = [poly.subs(x, 0) for poly in laguerre_pols]
    L0_deriv_vals = [poly.diff(x).subs(x, 0) for poly in laguerre_pols]
    
    for k in range(26):
        A[0, k] = L0_vals[k]
    b[0, 0] = 0
    
    for k in range(26):
        A[1, k] = L0_deriv_vals[k]
    b[1, 0] = 1
    
    for i in range(12):
        zi = z_rationals[i]
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_idx, 0] = 0
    
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None
    
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        try:
            q_at_minus = q_expr.subs(x, r - eps)
            q_at_plus = q_expr.subs(x, r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _score_numeric(z_list):
    """Fast numerical approximation of the score for optimization."""
    z = np.array(z_list, dtype=float)
    if np.any(z <= 0) or np.any(~np.isfinite(z)):
        return 1e10
    z_sorted = np.sort(z)
    if np.any(np.diff(z_sorted) < 1e-8):
        return 1e10
        
    alpha = -0.5
    max_deg = 50
    coeffs_L = _get_laguerre_coeffs()
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    for k in range(26):
        A[0, k] = coeffs_L[k][-1]
    b[0] = 0
    
    for k in range(26):
        if k == 0:
            A[1, k] = 0.0
        else:
            n = 2*k - 1
            alpha_deriv = 0.5
            L_deriv = genlaguerre(n, alpha_deriv)
            A[1, k] = -L_deriv.c[-1]
    b[1] = 1.0
    
    for i in range(12):
        zi = z[i]
        row_g = 2 + 2*i
        row_dg = 3 + 2*i
        
        for k in range(26):
            A[row_g, k] = eval_genlaguerre(2*k, alpha, zi)
            b[row_g] = 0.0
            
            if k == 0:
                A[row_dg, k] = 0.0
            else:
                n = 2*k - 1
                alpha_deriv = 0.5
                A[row_dg, k] = -eval_genlaguerre(n, alpha_deriv, zi)
            b[row_dg] = 0.0
            
    try:
        c_coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e10
        
    g_coeffs = np.zeros(max_deg + 1)
    for k in range(26):
        g_coeffs += c_coeffs[k] * coeffs_L[k]
        
    D_coeffs = np.array([1.0, 0.0])
    for zi in z:
        quad = np.array([1.0, -2.0*zi, zi**2])
        D_coeffs = np.convolve(D_coeffs, quad)
        
    try:
        q_coeffs, _ = np.polydiv(g_coeffs, D_coeffs)
    except Exception:
        return 1e10
        
    roots = np.roots(q_coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
    
    if not real_roots:
        return 0.0
        
    q_poly = np.poly1d(q_coeffs)
    q_prime_poly = np.polyder(q_poly)
    
    max_r = 0.0
    for r in real_roots:
        if abs(q_prime_poly(r)) > 1e-6:
            if r > max_r:
                max_r = r
                
    B = max(0.0, max_r)
    score = B / (2 * np.pi)
    return score

def _time_remaining(deadline):
    return deadline - time.time()

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    Uses numerical optimization to find a good configuration, then verifies with exact arithmetic.
    """
    deadline = time.time() + 570.0
    
    # Start from the zeros of the Laguerre polynomial L_12^(-0.5) as a structural initialization
    from numpy.polynomial.laguerre import lagval
    from scipy.special import roots_genlaguerre
    
    # Get zeros of L_12^(-0.5)
    x_zeros, _ = roots_genlaguerre(12, -0.5)
    z_initial = np.sort(x_zeros)
    
    # Ensure all are positive and distinct
    if np.any(z_initial <= 0) or np.any(np.diff(z_initial) < 1e-8):
        # Fallback to a simple spread
        z_initial = np.linspace(0.1, 10.0, 12)
    
    best_numeric_result = z_initial.copy()
    best_numeric_score = _score_numeric(z_initial)
    
    bounds = [(0.01, 50.0)] * 12
    
    def objective(z):
        return _score_numeric(z)
        
    # Use local optimization from the structural starting point
    try:
        res = minimize(objective, best_numeric_result, method='Powell', 
                       options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-10})
        if res.fun < best_numeric_score:
            best_numeric_score = res.fun
            best_numeric_result = res.x
    except Exception:
        pass
    
    # If time remains, try a few more local optimizations from perturbed starts
    if _time_remaining(deadline) > 150:
        for _ in range(5):
            z_pert = best_numeric_result * (1 + np.random.uniform(-0.05, 0.05, 12))
            z_pert = np.sort(z_pert)
            if np.all(z_pert > 0) and np.all(np.diff(z_pert) > 1e-6):
                try:
                    res = minimize(objective, z_pert, method='Nelder-Mead',
                                   options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_numeric_score:
                        best_numeric_score = res.fun
                        best_numeric_result = res.x
                except Exception:
                    pass
            if _time_remaining(deadline) < 100:
                break
    
    # Verify best candidate with exact arithmetic
    best_exact_score = float('inf')
    best_exact_z = None
    
    candidates = [best_numeric_result]
    
    for cand in candidates:
        if _time_remaining(deadline) < 90:
            break
        try:
            score = _compute_score_exact(cand)
            if score is not None and score < best_exact_score:
                best_exact_score = score
                best_exact_z = list(cand)
        except Exception:
            continue
            
    if best_exact_z is None:
        return [float(z) for z in z_initial]
        
    return best_exact_z
# EVOLVE-BLOCK-END

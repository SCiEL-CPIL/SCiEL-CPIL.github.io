"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre, genlaguerre

# EVOLVE-BLOCK-START

_L_coeffs_cache = None
_L_deriv_coeffs_cache = None

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

def _get_laguerre_deriv_coeffs():
    """Get coefficients of d/dx L_{2k}^{-1/2}(x) = -L_{2k-1}^{1/2}(x) for k >= 1, and 0 for k=0."""
    global _L_deriv_coeffs_cache
    if _L_deriv_coeffs_cache is None:
        _L_deriv_coeffs_cache = []
        max_deg = 50
        for k in range(26):
            if k == 0:
                _L_deriv_coeffs_cache.append(np.zeros(max_deg + 1))
            else:
                n = 2*k - 1
                alpha_deriv = 0.5
                L_deriv = genlaguerre(n, alpha_deriv)
                c = np.asarray(L_deriv.c, dtype=float)
                if len(c) < max_deg + 1:
                    c = np.pad(c, (0, max_deg + 1 - len(c)), mode='constant')
                _L_deriv_coeffs_cache.append(-c)  # Negative because L_n'(x) = -L_{n-1}^{(alpha+1)}(x)
    return _L_deriv_coeffs_cache

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
    coeffs_L_deriv = _get_laguerre_deriv_coeffs()
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    for k in range(26):
        A[0, k] = coeffs_L[k][-1]
    b[0] = 0
    
    for k in range(26):
        A[1, k] = coeffs_L_deriv[k][-1]
    b[1] = 1.0
    
    for i in range(12):
        zi = z[i]
        row_g = 2 + 2*i
        row_dg = 3 + 2*i
        
        for k in range(26):
            A[row_g, k] = eval_genlaguerre(2*k, alpha, zi)
            b[row_g] = 0.0
            A[row_dg, k] = -eval_genlaguerre(2*k - 1, 0.5, zi) if k > 0 else 0.0
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
        
    if len(q_coeffs) < 1:
        return 0.0
        
    roots = np.roots(q_coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
    
    if not real_roots:
        return 0.0
        
    q_poly = np.poly1d(q_coeffs)
    
    # Use sign change test like the exact version
    eps_num = 1e-15
    max_r = 0.0
    for r in real_roots:
        if r <= 0:
            continue
        try:
            q_minus = q_poly(r - eps_num)
            q_plus = q_poly(r + eps_num)
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
            
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
    
    from scipy.special import roots_genlaguerre
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Zeros of L_12^(-0.5)
    x_zeros_12, _ = roots_genlaguerre(12, -0.5)
    initializations.append(np.sort(x_zeros_12))
    
    # Zeros of L_12^(0) (standard Laguerre)
    x_zeros_std, _ = roots_genlaguerre(12, 0.0)
    initializations.append(np.sort(x_zeros_std))
    
    # Zeros of L_12^(0.5)
    x_zeros_pos, _ = roots_genlaguerre(12, 0.5)
    initializations.append(np.sort(x_zeros_pos))
    
    # Geometric spacing
    initializations.append(np.geomspace(0.1, 20.0, 12))
    
    # Logarithmic spacing
    initializations.append(np.logspace(-1, 1.3, 12))
    
    # Filter valid initializations
    valid_inits = []
    for z_init in initializations:
        z_init = np.array(z_init, dtype=float)
        if np.all(z_init > 0) and np.all(np.isfinite(z_init)):
            z_sorted = np.sort(z_init)
            if np.all(np.diff(z_sorted) > 1e-6):
                valid_inits.append(z_sorted)
    
    if not valid_inits:
        valid_inits = [np.linspace(0.1, 10.0, 12)]
    
    # Evaluate all initializations numerically
    best_numeric_result = None
    best_numeric_score = float('inf')
    
    for z_init in valid_inits:
        score = _score_numeric(z_init)
        if score < best_numeric_score:
            best_numeric_score = score
            best_numeric_result = z_init.copy()
    
    if best_numeric_result is None:
        return [float(z) for z in valid_inits[0]]
    
    def objective(z):
        return _score_numeric(z)
    
    # Multiple local optimizations from different starting points
    all_candidates = [best_numeric_result.copy()]
    
    # Powell from best initial
    try:
        res = minimize(objective, best_numeric_result, method='Powell', 
                       options={'maxiter': 300, 'xtol': 1e-12, 'ftol': 1e-12})
        if res.fun < best_numeric_score:
            best_numeric_score = res.fun
            best_numeric_result = res.x.copy()
            all_candidates.append(res.x.copy())
    except Exception:
        pass
    
    # Try Nelder-Mead from perturbed versions of best
    if _time_remaining(deadline) > 200:
        for trial in range(8):
            z_pert = best_numeric_result * (1 + np.random.uniform(-0.1, 0.1, 12))
            z_pert = np.sort(z_pert)
            if np.all(z_pert > 0) and np.all(np.diff(z_pert) > 1e-6):
                try:
                    res = minimize(objective, z_pert, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_numeric_score:
                        best_numeric_score = res.fun
                        best_numeric_result = res.x.copy()
                        all_candidates.append(res.x.copy())
                except Exception:
                    pass
            if _time_remaining(deadline) < 150:
                break
    
    # Also try from other good initializations
    if _time_remaining(deadline) > 250:
        for z_init in valid_inits[1:]:
            if _time_remaining(deadline) < 150:
                break
            try:
                res = minimize(objective, z_init, method='Powell', 
                               options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10})
                if res.fun < best_numeric_score:
                    best_numeric_score = res.fun
                    best_numeric_result = res.x.copy()
                    all_candidates.append(res.x.copy())
            except Exception:
                pass
    
    # Verify top candidates with exact arithmetic
    best_exact_score = float('inf')
    best_exact_z = None
    
    # Sort candidates by numeric score and verify top ones
    all_candidates.sort(key=lambda z: _score_numeric(z))
    
    for cand in all_candidates[:3]:  # Verify top 3
        if _time_remaining(deadline) < 120:
            break
        try:
            score = _compute_score_exact(cand)
            if score is not None and score < best_exact_score:
                best_exact_score = score
                best_exact_z = list(cand)
        except Exception:
            continue
            
    if best_exact_z is None:
        # Fallback: verify the best numeric candidate
        try:
            score = _compute_score_exact(best_numeric_result)
            if score is not None:
                best_exact_z = list(best_numeric_result)
        except Exception:
            pass
    
    if best_exact_z is None:
        return [float(z) for z in best_numeric_result]
        
    return best_exact_z
# EVOLVE-BLOCK-END

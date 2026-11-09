import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize
import time
import math

def _compute_score(z_list):
    """
    Compute the exact score B/(2*pi) for a given list of 12 double-root locations.
    Uses sympy for exact rational arithmetic.
    Returns B/(2*pi), or a large number if invalid.
    """
    import sympy as sp
    
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    
    # Build the generalized Laguerre polynomials L_0, L_2, ..., L_50
    laguerre_pols = []
    for k in range(26):
        d = 2 * k
        L_d = sp.genlaguerre(d, alpha)
        laguerre_pols.append(L_d)
    
    # Unknowns: c_0, c_1, ..., c_25
    c = sp.symbols('c0:26')
    
    # Build g(x) = sum_k c_k * L_{2k}(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    # g'(x)
    gp = sp.diff(g, x)
    
    # Constraints: g(0) = 0, g'(0) = 1
    constraints = [g.subs(x, 0), gp.subs(x, 0) - 1]
    
    # For each z_i: g(z_i) = 0, g'(z_i) = 0
    for z in z_list:
        z_r = sp.Rational(float(z))
        constraints.append(g.subs(x, z_r))
        constraints.append(gp.subs(x, z_r))
    
    # Solve the linear system
    A = sp.Matrix([sp.diff(constraint, c[k]) for constraint in constraints for k in range(26)])
    b = sp.Matrix([-constraint for constraint in constraints])
    
    try:
        sol = A.LUsolve(b)
    except Exception:
        return 1e10  # Singular system
    
    # Substitute solution into g
    g_sol = g.subs({c[k]: sol[k] for k in range(26)})
    
    # Build the denominator: x * prod_i (x - z_i)^2
    denom = x
    for z in z_list:
        z_r = sp.Rational(float(z))
        denom *= (x - z_r)**2
    
    # q(x) = g(x) / denom
    q = sp.Poly(g_sol, x) / sp.Poly(denom, x)
    
    if q.is_zero():
        return 0.0
    
    # Find real roots of q
    q_poly = sp.Poly(g_sol, x)
    denom_poly = sp.Poly(denom, x)
    q_exact = sp.cancel(q_poly / denom_poly)
    
    # Get numerator of the simplified rational function
    q_num = q_exact.as_expr()
    
    # Find real roots
    try:
        roots = sp.real_roots(q_num, x)
    except Exception:
        return 1e10
    
    # Count roots with sign change test
    counted_roots = []
    eps = sp.Rational(1, 10**198)
    
    for r in roots:
        if r <= 0:
            continue
        try:
            val_minus = q_num.subs(x, r - eps)
            val_plus = q_num.subs(x, r + eps)
            # Check for strictly opposite signs
            if val_minus * val_plus < 0:
                counted_roots.append(r)
        except Exception:
            continue
    
    if not counted_roots:
        return 0.0
    
    B = max(counted_roots)
    score = B / (2 * sp.pi)
    
    return float(score)

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Start from (k*pi)^2 for k=1..12 and use local optimization
    to minimize the exact score B/(2*pi).
    """
    start_time = time.time()
    deadline = 630.0  # seconds
    
    # Initial guess: (k*pi)^2 for k=1..12
    k_vals = np.arange(1, 13, dtype=np.float64)
    z_init = (k_vals * np.pi) ** 2
    
    # Also try alternative starting points
    # Option 1: (k*pi)^2
    # Option 2: Zeros of J_0 squared (asymptotic Laguerre zero distribution)
    from scipy.special import jn_zeros
    try:
        j0_zeros = jn_zeros(0, 12)
        z_alt = j0_zeros**2
    except:
        z_alt = z_init
    
    # Evaluate both starting points
    candidates = [z_init, z_alt]
    
    best_z = None
    best_score = float('inf')
    
    for z_start in candidates:
        if time.time() > deadline - 30:
            break
        
        try:
            score = _compute_score([float(z) for z in z_start])
            if score < best_score:
                best_score = score
                best_z = z_start.copy()
        except Exception:
            continue
    
    # If we have time, try local optimization
    if best_z is not None and time.time() < deadline - 60:
        try:
            def objective(z):
                if time.time() > deadline - 10:
                    return 1e10
                z_list = [float(zi) for zi in z]
                # Ensure positive and distinct
                if any(zi <= 0 for zi in z_list):
                    return 1e10
                if len(set(round(zi, 10) for zi in z_list)) < 12:
                    return 1e10
                try:
                    return _compute_score(z_list)
                except Exception:
                    return 1e10
            
            # Use Nelder-Mead for local optimization
            result = minimize(objective, best_z, method='Nelder-Mead',
                             options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-6})
            
            if result.fun < best_score:
                best_z = result.x
                best_score = result.fun
        except Exception:
            pass
    
    # Final validation
    if best_z is None:
        best_z = z_init
    
    # Ensure finite, > 0, pairwise distinct
    z_list = [float(z) for z in best_z]
    for i in range(len(z_list)):
        if not np.isfinite(z_list[i]) or z_list[i] <= 0:
            z_list[i] = (i + 1) * np.pi**2
    
    # Ensure distinct
    z_list.sort()
    for i in range(1, len(z_list)):
        if abs(z_list[i] - z_list[i-1]) < 1e-10:
            z_list[i] = z_list[i-1] + 1e-5
    
    return z_list

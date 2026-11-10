"""Refined program for Laguerre double roots (m=12, n=1).
Structural change: Use roots of derivative of L_13^(-1/2) as initial z_i.
"""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

# Cache for Laguerre polynomials to avoid re-computation if construct is called multiple times
# or if we want to reuse in helper functions. However, construct is called once per evaluation.
# But inside construct, we might call _compute_score_exact multiple times.
# So we should cache the polynomials globally or in a closure.

_LAGUERRE_CACHE = None
_X_SYMBOL = None

def _get_laguerre_pols():
    """Returns list of 26 Laguerre polynomials L_0^(-1/2) ... L_50^(-1/2) and x symbol."""
    global _LAGUERRE_CACHE, _X_SYMBOL
    if _LAGUERRE_CACHE is None:
        from sympy import symbols, Rational, laguerre
        _X_SYMBOL = symbols('x')
        alpha = Rational(-1, 2)
        _LAGUERRE_CACHE = []
        for d in range(0, 51, 2):
            try:
                lp = laguerre(d, alpha, _X_SYMBOL)
                _LAGUERRE_CACHE.append(lp)
            except Exception:
                return None, None
    return _LAGUERRE_CACHE, _X_SYMBOL

def _compute_score_exact(z_list, deadline):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic.
    Returns None if invalid or timeout."""
    from sympy import Rational, Poly, real_roots, pi as sympy_pi, Matrix
    
    if time.time() > deadline:
        return None
    
    # Validate z_list
    if len(z_list) != 12:
        return None
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return None
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-12:
                return None

    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    laguerre_pols, x = _get_laguerre_pols()
    if laguerre_pols is None:
        return None
    
    # Build matrix A (26x26) and vector b (26x1)
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Precompute derivatives for speed
    # A[0, k] = L_{2k}(0)
    # A[1, k] = L'_{2k}(0)
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[0, 0] = 0
    b[1, 0] = 1
    
    # For each z_i
    for i in range(12):
        zi = z_rationals[i]
        row_val = 2 + 2 * i
        row_der = 3 + 2 * i
        for k in range(26):
            A[row_val, k] = laguerre_pols[k].subs(x, zi)
            A[row_der, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_val, 0] = 0
        b[row_der, 0] = 0
    
    if time.time() > deadline:
        return None
    
    # Solve the linear system
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Build g(x) = sum c_k * L_{2k}(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    # q(x) = g(x) / denom via exact polynomial division
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        if time.time() > deadline:
            return None
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

def _is_valid_z(z_list):
    """Check if z_list is valid: finite, > 0, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-12:
                return False
    return True

def _get_structural_initial_z():
    """
    Get initial z_i as roots of derivative of L_13^(-1/2)(x).
    """
    from sympy import symbols, Rational, laguerre, Poly, real_roots
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    try:
        L13 = laguerre(13, alpha, x)
        dL13 = L13.diff(x)
        # Find roots of dL13
        roots = real_roots(dL13)
        # Filter for positive real roots
        pos_roots = [float(r) for r in roots if r > 0]
        
        # We expect 12 positive roots
        if len(pos_roots) == 12:
            return sorted(pos_roots)
        else:
            # Fallback to Bessel zeros if something went wrong
            return _get_bessel_fallback()
    except Exception:
        return _get_bessel_fallback()

def _get_bessel_fallback():
    """Fallback to Bessel J0 zeros squared."""
    z_bessel = []
    for k in range(1, 13):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            j0 = 2.4048255576957728
        else:
            j0 = x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
        z_bessel.append(j0**2)
    return z_bessel

def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 560.0  # 70s safety margin before 630s kill
    
    # 1. Structural Initialization
    initial_z = _get_structural_initial_z()
    
    if not _is_valid_z(initial_z):
        initial_z = [1.0 + i for i in range(12)] # Last resort fallback
    
    best_z = list(initial_z)
    best_score = float('inf')
    
    # 2. Evaluate Initial Configuration
    try:
        score = _compute_score_exact(best_z, deadline - 60)
        if score is not None:
            best_score = score
    except Exception:
        pass
    
    # 3. Local Refinement
    # We have limited time. Each eval takes ~10-60s.
    # We can afford ~5-10 evals.
    # Strategy: Coordinate descent with adaptive steps.
    
    if time.time() > deadline - 60:
        return best_z
    
    # Define search directions
    # We'll try perturbing each z_i individually.
    # Start with small steps, expand if improving, shrink if not.
    
    step_sizes = [0.1, 0.5, 1.0, 2.0]
    current_z = list(best_z)
    current_score = best_score
    
    # Iterate until time runs out
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        for i in range(12):
            if time.time() > deadline - 60:
                break
            # Try different step sizes
            for step in step_sizes:
                for sign in [1, -1]:
                    if time.time() > deadline - 30:
                        break
                    candidate = list(current_z)
                    candidate[i] = current_z[i] * (1 + sign * step)
                    if candidate[i] <= 0:
                        continue
                    if not _is_valid_z(candidate):
                        continue
                    
                    try:
                        score = _compute_score_exact(candidate, deadline - 30)
                        if score is not None and score < current_score:
                            current_score = score
                            current_z = candidate
                            improved = True
                            best_z = list(current_z)
                            best_score = score
                            # If improved, we can break out of step loop for this i
                            # and restart the i loop to re-evaluate neighbors
                            break
                    except Exception:
                        continue
                if improved:
                    break
    
    return best_z

# EVOLVE-BLOCK-END

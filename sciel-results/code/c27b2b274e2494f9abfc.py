"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _bessel_zero_J1(k):
    """Approximate k-th zero of J_1 using asymptotic expansion."""
    x = (k + 0.25) * math.pi
    if x < 0.1:
        return 3.8317059702075125  # j_{1,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix
    from sympy.polys.orthopolys import laguerre as _laguerre_gen
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    def L_gen_explicit(n, alpha_val, x_sym):
        from sympy import binomial
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / math.factorial(k)
        return result
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(L_gen_explicit(d, alpha, x))
    
    # Build system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^(alpha)(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 12*2 = 26 equations, 26 unknowns c_0..c_25
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[1, 0] = 1
    
    # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
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
        q_poly, _ = g_poly.div(denom_poly)
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

def _time_remaining(deadline):
    return deadline - time.time()

def _validate_z_list(z_list):
    """Check that z_list is valid: finite, > 0, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-10:
                return False
    return True

def _try_local_search(base_z, deadline, best_score, best_z):
    """Try local perturbations around base_z, update best if improved."""
    # Try scaling the entire configuration
    scale_factors = [0.8, 0.9, 1.1, 1.2]
    for sf in scale_factors:
        if _time_remaining(deadline) < 120:
            break
        candidate = [z * sf for z in base_z]
        if not _validate_z_list(candidate):
            continue
        score = _compute_score_exact(candidate)
        if score is not None and score < best_score:
            best_score = score
            best_z = candidate
    
    # Try individual perturbations in log-space
    for i in range(12):
        if _time_remaining(deadline) < 120:
            break
        for delta_log in [0.05, -0.05, 0.1, -0.1]:
            if _time_remaining(deadline) < 120:
                break
            candidate = list(best_z)
            candidate[i] = best_z[i] * math.exp(delta_log)
            if not _validate_z_list(candidate):
                continue
            score = _compute_score_exact(candidate)
            if score is not None and score < best_score:
                best_score = score
                best_z = candidate
    
    return best_score, best_z

def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel-zero-inspired parameterization."""
    deadline = time.time() + 580.0  # 630s kill, keep 50s margin for safety
    
    # Generate multiple candidate initial configurations
    candidates = []
    
    # 1. J_0 zeros squared
    z_j0 = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j0):
        candidates.append(z_j0)
    
    # 2. J_1 zeros squared
    z_j1 = [_bessel_zero_J1(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j1):
        candidates.append(z_j1)
    
    # 3. Geometric progression
    geo_ratio = 1.5
    z_geo = [2.0 * geo_ratio**i for i in range(12)]
    if _validate_z_list(z_geo):
        candidates.append(z_geo)
    
    # 4. Arithmetic progression
    z_arith = [2.0 + 3.0*i for i in range(12)]
    if _validate_z_list(z_arith):
        candidates.append(z_arith)
    
    # 5. J_0 zeros (not squared)
    z_j0_raw = [_bessel_zero_J0(i) for i in range(1, 13)]
    if _validate_z_list(z_j0_raw):
        candidates.append(z_j0_raw)
    
    # 6. Scaled J_0 squared
    z_j0_scaled = [z * 0.8 for z in z_j0]
    if _validate_z_list(z_j0_scaled):
        candidates.append(z_j0_scaled)
    
    z_j0_scaled2 = [z * 1.2 for z in z_j0]
    if _validate_z_list(z_j0_scaled2):
        candidates.append(z_j0_scaled2)
    
    # Evaluate each candidate and keep the best
    best_score = float('inf')
    best_z = candidates[0] if candidates else [4.0 * (i+1) for i in range(12)]
    
    for cand in candidates:
        if _time_remaining(deadline) < 120:
            break
        try:
            score = _compute_score_exact(cand)
            if score is not None and score < best_score:
                best_score = score
                best_z = cand
        except Exception:
            continue
    
    # Local search from the best configuration
    if _time_remaining(deadline) > 120:
        best_score, best_z = _try_local_search(best_z, deadline, best_score, best_z)
    
    # Final validation
    if not _validate_z_list(best_z):
        best_z = [4.0 * (i+1) for i in range(12)]
    
    return best_z
# EVOLVE-BLOCK-END

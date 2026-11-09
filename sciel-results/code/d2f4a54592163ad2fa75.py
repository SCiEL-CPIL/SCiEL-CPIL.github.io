"""Rewritten program: spectral/orthogonal-node inspired placement with fast local polish."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _bessel_zero_J1(k):
    """Approximate k-th zero of J_1 using asymptotic expansion."""
    x = (k + 0.25) * math.pi
    if x < 0.1:
        return 3.8317059702075125
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, laguerre

    z_rationals = sorted(Rational(float(z)) for z in z_list)

    x = symbols('x')
    alpha = Rational(-1, 2)

    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(laguerre(d, x, alpha))

    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)

    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0

    for k in range(26):
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
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

def _laguerre_nodes_weighted(n, alpha, weight_poly_coeffs):
    """
    Generate n positive nodes that are roots of a modified Laguerre-related polynomial.
    weight_poly_coeffs: list of coefficients for a low-degree polynomial h(x) (ascending powers).
    We form p(x) = L_n^(alpha)(x) * h(x) and find its positive roots via numpy roots.
    This is a spectral/orthogonal-node inspired placement.
    """
    from numpy.polynomial import laguerre as nplag
    # Build L_n^(alpha)(x) coefficients (ascending)
    # numpy laguerre.Laguerre(n, alpha) gives a Laguerre basis object
    L = nplag.Laguerre(n, alpha)
    L_coeffs = L.coef  # ascending
    # Multiply by h(x)
    h_coeffs = np.array(weight_poly_coeffs, dtype=float)
    p_coeffs = np.convolve(L_coeffs, h_coeffs)
    # Find roots
    roots = np.roots(p_coeffs[::-1])  # np.roots expects descending
    # Filter positive real roots
    pos_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 0:
            pos_roots.append(r.real)
    pos_roots = sorted(pos_roots)
    # We need exactly n positive roots; if not, fall back to standard Laguerre nodes
    if len(pos_roots) < n:
        # Use standard Laguerre-Gauss nodes for alpha
        x_nodes = nplag.laggauss(n, alpha)
        return list(x_nodes)
    # Return the first n positive roots (should be n if h has no positive roots that cancel)
    return pos_roots[:n]

def _standard_laguerre_nodes(n, alpha):
    """Standard Laguerre-Gauss nodes for parameter alpha."""
    from numpy.polynomial import laguerre as nplag
    x_nodes = nplag.laggauss(n, alpha)
    return list(x_nodes)

def _local_search(z_list, deadline, max_iter=50, step=0.1):
    """
    Simple coordinate descent local search to minimize the score.
    Each evaluation is expensive, so we limit iterations.
    """
    best_z = list(z_list)
    best_score = _compute_score_exact(best_z)
    if best_score is None:
        return best_z, float('inf')
    
    for _ in range(max_iter):
        if _time_remaining(deadline) < 120:
            break
        improved = False
        for i in range(12):
            if _time_remaining(deadline) < 120:
                break
            for delta in [-step, step]:
                new_z = list(best_z)
                new_z[i] += delta
                if not _validate_z_list(new_z):
                    continue
                try:
                    score = _compute_score_exact(new_z)
                    if score is not None and score < best_score - 1e-12:
                        best_z = new_z
                        best_score = score
                        improved = True
                        break
                except Exception:
                    continue
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    return best_z, best_score

def construct():
    """Return 12 double-root locations z_1..z_12 using spectral/orthogonal-node inspired placement."""
    deadline = time.time() + 550.0
    alpha = -0.5
    
    candidates = []
    
    # 1. Standard Laguerre-Gauss nodes for alpha = -1/2
    try:
        z_lag = _standard_laguerre_nodes(12, alpha)
        if _validate_z_list(z_lag):
            candidates.append(z_lag)
    except Exception:
        pass
    
    # 2. J_0 zeros squared
    z_j0 = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j0):
        candidates.append(z_j0)
    
    # 3. Modified Laguerre nodes with h(x) = 1 + 0.1*x
    try:
        z_mod = _laguerre_nodes_weighted(12, alpha, [1.0, 0.1])
        if _validate_z_list(z_mod):
            candidates.append(z_mod)
    except Exception:
        pass
    
    # 4. J_1 zeros squared
    z_j1 = [_bessel_zero_J1(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j1):
        candidates.append(z_j1)
    
    # 5. Geometric progression
    geo_ratio = 1.5
    z_geo = [2.0 * geo_ratio**i for i in range(12)]
    if _validate_z_list(z_geo):
        candidates.append(z_geo)
    
    # 6. Arithmetic progression
    z_arith = [2.0 + 3.0*i for i in range(12)]
    if _validate_z_list(z_arith):
        candidates.append(z_arith)
    
    default_z = [4.0 * (i+1) for i in range(12)]
    if not candidates:
        candidates.append(default_z)
    
    best_score = float('inf')
    best_z = candidates[0]
    
    # Evaluate the first few candidates if time permits
    for i, cand in enumerate(candidates[:4]):
        if _time_remaining(deadline) < 150:
            break
        try:
            score = _compute_score_exact(cand)
            if score is not None and score < best_score:
                best_score = score
                best_z = list(cand)
        except Exception:
            pass
    
    # If we have time, do a local search from the best candidate
    if _time_remaining(deadline) > 180:
        try:
            best_z, best_score = _local_search(best_z, deadline, max_iter=30, step=0.1)
        except Exception:
            pass
    
    # Final validation
    if not _validate_z_list(best_z):
        best_z = default_z
    
    return best_z
# EVOLVE-BLOCK-END

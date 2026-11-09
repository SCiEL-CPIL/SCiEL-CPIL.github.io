"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    # j_{0,k} ≈ (k - 1/4)*π - 1/(8*(k-1/4)*π) + 9/(128*(k-1/4)^3*π^3) - 225/(3072*(k-1/4)^5*π^5)
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix
    from sympy.polys.orthopolys import laguerre as _laguerre_gen
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    # Use the explicit formula for generalized Laguerre
    def L_gen(n, alpha_val, x_sym):
        """Generalized Laguerre polynomial L_n^(alpha)(x) via explicit sum."""
        from sympy import binomial
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result += coeff * x_sym**k / math.factorial(k)
        return result
    
    # Actually, use sympy's built-in if available, otherwise explicit
    from sympy import binomial
    def L_gen_explicit(n, alpha_val, x_sym):
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
    
    # Build matrix A and vector b
    # Row 0: g(0) = sum c_k * L_{2k}(0) = 0
    # Row 1: g'(0) = sum c_k * L'_{2k}(0) = 1
    # Rows 2..25: g(z_i) = 0, g'(z_i) = 0
    
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
        # g(zi) = 0
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        # g'(zi) = 0
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_idx, 0] = 0
    
    # Solve
    try:
        c = A.LUsolve(b)
    except Exception:
        return None  # singular system
    
    # Build g(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    # q(x) = g(x) / denom via exact polynomial division
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, _ = g_poly.div(denom_poly)
    except Exception:
        return None
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    # 1e-198 as rational
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

def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel-zero-inspired parameterization."""
    deadline = time.time() + 570.0  # 630s kill, keep 60s margin for safety
    
    # Compute Bessel zeros of J_0 for k=1..12
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # Ensure positive, finite, distinct
    result = list(z_bessel)
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            result = [4.0 * (i + 1) for i in range(12)]
            break
    
    # Check pairwise distinct
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [4.0 * (i + 1) for i in range(12)]
                break
    
    # If time permits, try to refine with small perturbations
    remaining = _time_remaining(deadline)
    if remaining > 120:  # Need at least ~120s for one evaluation
        try:
            best_score = _compute_score_exact(result)
            if best_score is not None:
                # Try small perturbations of each z_i
                for i in range(12):
                    if _time_remaining(deadline) < 90:
                        break
                    for delta_frac in [0.01, -0.01, 0.02, -0.02]:
                        if _time_remaining(deadline) < 90:
                            break
                        candidate = list(result)
                        candidate[i] = result[i] * (1 + delta_frac)
                        if candidate[i] <= 0:
                            continue
                        # Check distinctness
                        valid = True
                        for a in range(12):
                            for b_idx in range(a+1, 12):
                                if abs(candidate[a] - candidate[b_idx]) < 1e-10:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        score = _compute_score_exact(candidate)
                        if score is not None and score < best_score:
                            best_score = score
                            result = candidate
        except Exception:
            pass
    
    return result
# EVOLVE-BLOCK-END

"""Rewritten program: Spectral Interpolation with Laguerre Orthogonality Constraints.

Strategy:
  - Use the 12 positive zeros of the generalized Laguerre polynomial L_12^(alpha)(x)
    with alpha = -1/2 as the initial double-root locations. These nodes are the
    Gaussian quadrature nodes for the weight w(x) = exp(-x) x^{-1/2}, which
    naturally minimizes the spectral radius of interpolation in this basis.
  - Refine by trying small multiplicative perturbations to reduce B/(2*pi).
"""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _laguerre_zeros(n, alpha):
    """
    Compute the n positive zeros of the generalized Laguerre polynomial L_n^(alpha)(x).
    Uses the companion matrix eigenvalue approach via numpy for speed and accuracy.
    """
    from numpy.polynomial import laguerre as nplag
    # nplag.laggauss gives zeros and weights for alpha=0.
    # For general alpha, use the recurrence to build the companion matrix.
    # L_n^(alpha)(x) satisfies a three-term recurrence:
    # (n+1) L_{n+1}^(alpha)(x) = (2n + alpha + 1 - x) L_n^(alpha)(x) - (n + alpha) L_{n-1}^(alpha)(x)
    # Build the companion matrix for the monic version.
    # Alternatively, use sympy for exactness, but numpy is faster for initial guess.
    
    # Use the recurrence to form the symmetric tridiagonal matrix whose eigenvalues
    # are the zeros of L_n^(alpha).
    # The zeros of L_n^(alpha) are the eigenvalues of the matrix:
    #   T_{i,i} = 2i + alpha + 1  (for i=0..n-1, 0-indexed)
    #   T_{i,i+1} = T_{i+1,i} = sqrt((i+1)(i+1+alpha))  (for i=0..n-2)
    # Wait, that's for the orthogonal polynomial associated with the measure.
    # Actually, the zeros of L_n^(alpha) are the eigenvalues of the Jacobi matrix:
    #   J_{i,i} = 2i + alpha + 1
    #   J_{i,i+1} = J_{i+1,i} = sqrt((i+1)(i+1+alpha))
    # for i = 0, ..., n-1.
    
    n_int = int(n)
    J = np.zeros((n_int, n_int))
    for i in range(n_int):
        J[i, i] = 2*i + alpha + 1
        if i + 1 < n_int:
            val = math.sqrt((i+1)*(i+1+alpha))
            J[i, i+1] = val
            J[i+1, i] = val
    
    eigvals = np.linalg.eigvalsh(J)
    return sorted(eigvals.tolist())


def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, binomial, factorial
    from sympy import nsimplify
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / factorial(k)
        return result
    
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(L_gen_explicit(d, alpha, x))
    
    # Build matrix A and vector b
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


def construct():
    """Return 12 double-root locations z_1..z_12 using Laguerre-node-inspired parameterization."""
    deadline = time.time() + 560.0  # Keep margin for safety
    
    # Initial guess: zeros of L_12^(-1/2)(x)
    alpha = -0.5
    z_lag = _laguerre_zeros(12, alpha)
    
    # Ensure positive, finite, distinct
    result = [float(z) for z in z_lag]
    valid = True
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            valid = False
            break
    if valid:
        for i in range(12):
            for j in range(i+1, 12):
                if abs(result[i] - result[j]) < 1e-12:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback: Bessel zeros squared
        def _bessel_zero_J0(k):
            x = (k - 0.25) * math.pi
            if x < 0.1:
                return 2.4048255576957728
            return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
        result = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # Evaluate initial score
    best_score = None
    remaining = _time_remaining(deadline)
    if remaining > 120:
        try:
            best_score = _compute_score_exact(result)
        except Exception:
            best_score = None
    
    if best_score is not None:
        # Try small multiplicative perturbations
        remaining = _time_remaining(deadline)
        if remaining > 150:
            # Try perturbing each z_i by small factors
            deltas = [0.005, -0.005, 0.01, -0.01, 0.02, -0.02]
            for i in range(12):
                if _time_remaining(deadline) < 100:
                    break
                for d in deltas:
                    if _time_remaining(deadline) < 100:
                        break
                    candidate = list(result)
                    candidate[i] = result[i] * (1.0 + d)
                    if candidate[i] <= 0:
                        continue
                    # Check distinctness
                    ok = True
                    for a in range(12):
                        for b_idx in range(a+1, 12):
                            if abs(candidate[a] - candidate[b_idx]) < 1e-10:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    try:
                        s = _compute_score_exact(candidate)
                        if s is not None and s < best_score:
                            best_score = s
                            result = candidate
                    except Exception:
                        pass
    
    return result

# EVOLVE-BLOCK-END

import time
import math
import numpy as np
import scipy.special
import scipy.linalg

def _compute_score_fast(z_list):
    """Compute B/(2*pi) for given z_list using numerical methods."""
    z_sorted = sorted([float(z) for z in z_list])
    if len(z_sorted) != 12:
        return None
    
    # Build Laguerre polynomials L_d^{(-1/2)}(x) for d = 0, 2, 4, ..., 50
    def L_gen(d, x_val):
        return scipy.special.eval_genlaguerre(d, -0.5, x_val)
    
    def L_gen_deriv(d, x_val):
        # Derivative of L_n^alpha(x) = -L_{n-1}^{alpha+1}(x)
        if d == 0:
            return 0.0
        return -scipy.special.eval_genlaguerre(d-1, 0.5, x_val)
    
    # Build matrix A (26x26) and vector b (26x1)
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Row 0: g(0) = 0
    for k in range(26):
        d = 2*k
        A[0, k] = L_gen(d, 0.0)
    b[0] = 0.0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        d = 2*k
        A[1, k] = L_gen_deriv(d, 0.0)
    b[1] = 1.0
    
    # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
    for i in range(12):
        zi = z_sorted[i]
        row_idx = 2 + 2*i
        for k in range(26):
            d = 2*k
            A[row_idx, k] = L_gen(d, zi)
        b[row_idx] = 0.0
        row_idx = 3 + 2*i
        for k in range(26):
            d = 2*k
            A[row_idx, k] = L_gen_deriv(d, zi)
        b[row_idx] = 0.0
    
    # Solve linear system
    try:
        c = scipy.linalg.solve(A, b, assume_a='gen')
    except Exception:
        return None
    
    # Build g(x) polynomial coefficients
    def laguerre_coeffs(n, alpha):
        """Return coefficients of L_n^alpha(x) as list [c_0, c_1, ..., c_n]"""
        coeffs = [0.0] * (n + 1)
        for k in range(n + 1):
            a = n + alpha
            b_val = n - k
            if b_val < 0 or a - b_val < 0:
                coeff_val = 0.0
            else:
                from math import lgamma
                log_binom = lgamma(a + 1) - lgamma(b_val + 1) - lgamma(a - b_val + 1)
                binom_val = math.exp(log_binom)
                coeff_val = binom_val * ((-1)**k) / math.factorial(k)
            coeffs[k] = coeff_val
        return coeffs
    
    g_poly = [0.0] * 51
    for k in range(26):
        d = 2*k
        lc = laguerre_coeffs(d, -0.5)
        for j in range(d + 1):
            g_poly[j] += c[k] * lc[j]
    
    # Build denominator: x * prod(x - z_i)^2
    denom = [0.0, 1.0]  # x
    for zi in z_sorted:
        sq = [zi*zi, -2*zi, 1.0]
        new_denom = [0.0] * (len(denom) + len(sq) - 1)
        for i in range(len(denom)):
            for j in range(len(sq)):
                new_denom[i+j] += denom[i] * sq[j]
        denom = new_denom
    
    # Polynomial long division
    def poly_div(dividend, divisor):
        dividend = list(dividend)
        divisor = list(divisor)
        while len(dividend) > 0 and abs(dividend[-1]) < 1e-15:
            dividend.pop()
        while len(divisor) > 0 and abs(divisor[-1]) < 1e-15:
            divisor.pop()
        if len(dividend) < len(divisor):
            return [0.0], list(dividend)
        quotient = [0.0] * (len(dividend) - len(divisor) + 1)
        remainder = list(dividend)
        for i in range(len(quotient) - 1, -1, -1):
            if abs(divisor[-1]) < 1e-15:
                break
            factor = remainder[i + len(divisor) - 1] / divisor[-1]
            quotient[i] = factor
            for j in range(len(divisor)):
                remainder[i + j] -= factor * divisor[j]
        while len(remainder) > 0 and abs(remainder[-1]) < 1e-10:
            remainder.pop()
        return quotient, remainder
    
    try:
        q_poly, rem = poly_div(g_poly, denom)
    except Exception:
        return None
    
    # Find real roots of q_poly
    try:
        roots = np.roots(q_poly)
    except Exception:
        return None
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    if not real_roots:
        return 0.0
    
    def q_func(x_val):
        result = 0.0
        for i, coeff in enumerate(q_poly):
            result += coeff * (x_val ** i)
        return result
    
    eps = 1e-198
    max_r = 0.0
    for r in real_roots:
        try:
            q_minus = q_func(r - eps)
            q_plus = q_func(r + eps)
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0.0, max_r)
    score = B / (2 * math.pi)
    return score

def _bessel_zero_J0(k):
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def construct():
    deadline = time.time() + 570.0
    
    # Try several candidate z_i sets
    candidates = []
    
    # Candidate 1: Bessel zeros squared
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    candidates.append(z_bessel)
    
    # Candidate 2: Squares of integers
    z_squares = [float(i*i) for i in range(1, 13)]
    candidates.append(z_squares)
    
    # Candidate 3: Laguerre roots
    try:
        roots_lag = scipy.special.roots_genlaguerre(12, -0.5)[0]
        z_lag = sorted([float(r) for r in roots_lag if r > 0])
        if len(z_lag) == 12:
            candidates.append(z_lag)
    except Exception:
        pass
    
    # Candidate 4: Geometric progression
    z_geo = [1.0 * (2.0 ** i) for i in range(12)]
    candidates.append(z_geo)
    
    # Candidate 5: Larger spread
    z_large = [float(i*10) for i in range(1, 13)]
    candidates.append(z_large)
    
    best_score = float('inf')
    best_z = z_bessel
    
    for z_candidate in candidates:
        if time.time() > deadline - 30:
            break
        try:
            score = _compute_score_fast(z_candidate)
            if score is not None and score < best_score:
                best_score = score
                best_z = z_candidate
        except Exception:
            continue
    
    return best_z

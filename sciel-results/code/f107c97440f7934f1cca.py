import time
import math
import numpy as np
import scipy.special
import sympy

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, binomial
    
    # Ensure z_list is sorted and distinct
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    # Check for duplicates
    if len(set(z_rationals)) != 12:
        return None
        
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    # Using the explicit formula to avoid potential issues with sympy's laguerre for generalized alpha
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            # binomial(n + alpha, n - k) * (-1)^k / k! * x^k
            # Note: binomial(n + alpha, n - k) = binomial(n + alpha, k)
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / sympy.factorial(k)
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
        # Check singularity quickly? LUsolve will raise if singular.
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
        
    # Optimize polynomial domain for real_roots
    try:
        q_poly = q_poly.set_domain('QQ')
    except Exception:
        pass
        
    try:
        real_roots_list = real_roots(q_poly)
    except Exception:
        return None
        
    eps = Rational(1, 10)**198
    max_r = 0
    for r in real_roots_list:
        try:
            q_at_minus = q_poly.eval(r - eps)
            q_at_plus = q_poly.eval(r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
            
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def construct():
    start_time = time.time()
    deadline = 600  # 10 minutes
    
    # Candidate 1: Roots of L_12^(-1/2)(x)
    # These are the natural nodes for the Laguerre expansion of order -1/2
    try:
        z_cand1 = scipy.special.roots_genlaguerre(12, -0.5)[0]
        z_cand1 = [float(r) for r in z_cand1 if r > 0]
    except Exception:
        z_cand1 = [float(i) for i in range(1, 13)]
        
    # Candidate 2: Roots of L_13^(-1/2)(x), taking the first 12
    # This might spread the roots differently
    try:
        z_cand2 = scipy.special.roots_genlaguerre(13, -0.5)[0]
        z_cand2 = [float(r) for r in z_cand2 if r > 0][:12]
    except Exception:
        z_cand2 = [float(i + 0.5) for i in range(1, 13)]
        
    # Candidate 3: Bessel zeros J_{-1/4} squared / 4
    # Based on the asymptotic relation L_n^(-1/2)(x) ~ J_{-1/4}(2sqrt(x))
    try:
        j_zeros = scipy.special.jn_zeros(-0.25, 12)
        z_cand3 = [float(j**2 / 4.0) for j in j_zeros]
    except Exception:
        z_cand3 = z_cand1

    best_score = float('inf')
    best_z = z_cand1
    
    # Evaluate Candidate 1
    if time.time() - start_time < deadline - 60:
        score1 = _compute_score_exact(z_cand1)
        if score1 is not None and score1 < best_score:
            best_score = score1
            best_z = z_cand1

    # Evaluate Candidate 2
    if time.time() - start_time < deadline - 60:
        score2 = _compute_score_exact(z_cand2)
        if score2 is not None and score2 < best_score:
            best_score = score2
            best_z = z_cand2

    # Evaluate Candidate 3
    if time.time() - start_time < deadline - 60:
        score3 = _compute_score_exact(z_cand3)
        if score3 is not None and score3 < best_score:
            best_score = score3
            best_z = z_cand3

    return best_z

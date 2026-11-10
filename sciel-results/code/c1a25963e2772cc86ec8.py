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
    
    # Check for distinct positive values
    for i in range(11):
        if abs(z_sorted[i+1] - z_sorted[i]) < 1e-10:
            return None
    if any(z <= 0 for z in z_sorted):
        return None
    
    # Build Laguerre polynomials L_d^{(-1/2)}(x) for d = 0, 2, 4, ..., 50
    def L_gen(d, x_val):
        return scipy.special.eval_genlaguerre(d, -0.5, x_val)
    
    def L_gen_deriv(d, x_val):
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
    
    # Solve the system
    try:
        cond = np.linalg.cond(A)
        if cond > 1e15:
            return None
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
    denom_coeffs = np.array([1.0, 0.0])  # x
    for zi in z_sorted:
        sq = np.array([1.0, -2.0*zi, zi*zi])
        denom_coeffs = np.convolve(denom_coeffs, sq)
    
    # Polynomial long division
    try:
        q_poly_arr, rem_arr = np.polydiv(g_poly[::-1], denom_coeffs)
        q_poly = q_poly_arr[::-1].tolist()
        
        if np.max(np.abs(rem_arr)) > 1e-8 * max(1.0, np.max(np.abs(q_poly_arr))):
            return None
    except Exception:
        return None
    
    # Find real roots of q_poly
    try:
        q_poly_np = np.array(q_poly[::-1])
        while len(q_poly_np) > 1 and abs(q_poly_np[0]) < 1e-15:
            q_poly_np = q_poly_np[1:]
        
        if len(q_poly_np) <= 1:
            return 0.0
            
        roots = np.roots(q_poly_np)
    except Exception:
        return None
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            real_roots.append(r.real)
    
    if not real_roots:
        return 0.0
    
    # Evaluate q using Horner's method
    def q_func(x_val):
        result = 0.0
        for coeff in reversed(q_poly):
            result = result * x_val + coeff
        return result
    
    # Use a small epsilon for sign check
    eps = 1e-10
    max_r = 0.0
    for r in real_roots:
        if r <= 0:
            continue
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
    """Compute the k-th zero of J0."""
    zeros = [2.4048255576957728, 5.5200781102863106, 8.653727912911012,
             11.791534439014356, 14.930917708487787, 18.071063967910928,
             21.211636629971378, 24.352471530849282, 27.49347913204042,
             30.63460646843148, 33.77582021366707, 36.91709835369574,
             40.05815260865461, 43.19854537180558, 46.33890017968481,
             49.47922420577383, 52.61952291350434, 55.75979996909348,
             58.89995896344063, 62.04000344896869, 65.18003687008761,
             68.32006264136025, 71.46008317726022, 74.60009990237222,
             77.74011423128105, 80.88012747857258, 84.02013994883273,
             87.16015184764746, 90.30016337951277, 93.44017463892466]
    if k <= len(zeros):
        return zeros[k-1]
    x = (k - 0.25) * math.pi
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _local_optimize(z0, max_iter=80, step_init=0.1, deadline=None):
    """Local optimization using coordinate descent with adaptive step size."""
    z = sorted([float(z) for z in z0])
    best_score = _compute_score_fast(z)
    if best_score is None:
        return z, float('inf')
    
    step = step_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 30:
            break
        improved = False
        for i in range(12):
            # Try moving z[i] with multiple step sizes
            for delta in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1,
                         step*2, -step*2]:
                z_new = list(z)
                z_new[i] += delta
                z_new_sorted = sorted(z_new)
                valid = True
                for j in range(11):
                    if abs(z_new_sorted[j+1] - z_new_sorted[j]) < 1e-8:
                        valid = False
                        break
                if not valid or any(z <= 0 for z in z_new_sorted):
                    continue
                score_new = _compute_score_fast(z_new_sorted)
                if score_new is not None and score_new < best_score - 1e-12:
                    z = z_new_sorted
                    best_score = score_new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.7
            if step < 1e-7:
                break
    return z, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Try several candidate z_i sets
    candidates = []
    
    # Candidate 1: Bessel zeros squared
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    candidates.append(z_bessel)
    
    # Candidate 2: Squares of integers
    z_squares = [float(i*i) for i in range(1, 13)]
    candidates.append(z_squares)
    
    # Candidate 3: Laguerre roots L_12^(-1/2)
    try:
        roots_lag = scipy.special.roots_genlaguerre(12, -0.5)[0]
        z_lag = sorted([float(r) for r in roots_lag if r > 0])
        if len(z_lag) == 12:
            candidates.append(z_lag)
    except Exception:
        pass
    
    # Candidate 4: Roots of derivative of L_13^(-1/2)
    try:
        roots_lag_deriv = scipy.special.roots_genlaguerre(12, 0.5)[0]
        z_lag_deriv = sorted([float(r) for r in roots_lag_deriv if r > 0])
        if len(z_lag_deriv) == 12:
            candidates.append(z_lag_deriv)
    except Exception:
        pass
    
    # Candidate 5: Geometric progression
    z_geo = [1.0 * (2.0 ** i) for i in range(12)]
    candidates.append(z_geo)
    
    # Candidate 6: Larger spread
    z_large = [float(i*10) for i in range(1, 13)]
    candidates.append(z_large)
    
    # Candidate 7: Smaller values
    z_small = [float(i * 0.5) for i in range(1, 13)]
    candidates.append(z_small)
    
    # Candidate 8: Interlaced with Laguerre roots
    try:
        roots_lag13 = scipy.special.roots_genlaguerre(13, -0.5)[0]
        z_lag13 = sorted([float(r) for r in roots_lag13 if r > 0])
        if len(z_lag13) == 13:
            candidates.append(z_lag13[:12])
            candidates.append(z_lag13[1:])
    except Exception:
        pass
    
    # Candidate 9: Power law with exponent 1.5
    z_power = [float(i ** 1.5) for i in range(1, 13)]
    candidates.append(z_power)
    
    # Candidate 10: Logarithmic spacing
    z_log = [math.exp(i * 0.3) for i in range(1, 13)]
    candidates.append(z_log)
    
    # Candidate 11: Roots of L_14^(-1/2) taking 12
    try:
        roots_lag14 = scipy.special.roots_genlaguerre(14, -0.5)[0]
        z_lag14 = sorted([float(r) for r in roots_lag14 if r > 0])
        if len(z_lag14) == 14:
            candidates.append(z_lag14[:12])
            candidates.append(z_lag14[2:])
    except Exception:
        pass
    
    # Candidate 12: Scaled Bessel zeros
    for scale in [0.3, 0.5, 0.7, 1.5, 2.0]:
        z_bessel_scaled = [(_bessel_zero_J0(i)**2) * scale for i in range(1, 13)]
        candidates.append(z_bessel_scaled)
    
    # Candidate 13: Mixed Laguerre roots
    try:
        roots_l11 = scipy.special.roots_genlaguerre(11, 0.5)[0]
        roots_l12 = scipy.special.roots_genlaguerre(12, 0.5)[0]
        z_l11 = sorted([float(r) for r in roots_l11 if r > 0])
        z_l12 = sorted([float(r) for r in roots_l12 if r > 0])
        if len(z_l11) == 11 and len(z_l12) == 12:
            z_mixed = []
            i1, i2 = 0, 0
            while len(z_mixed) < 12:
                if i1 < len(z_l11) and (i2 >= len(z_l12) or z_l11[i1] < z_l12[i2]):
                    z_mixed.append(z_l11[i1])
                    i1 += 1
                else:
                    z_mixed.append(z_l12[i2])
                    i2 += 1
            candidates.append(z_mixed)
    except Exception:
        pass
    
    # Candidate 14: Bessel zeros with different scaling
    z_bessel_lin = [float(_bessel_zero_J0(i)) for i in range(1, 13)]
    candidates.append(z_bessel_lin)
    
    # Candidate 15: Cubic growth
    z_cubic = [float(i ** 2.5) for i in range(1, 13)]
    candidates.append(z_cubic)
    
    # Candidate 16: Very small values
    z_tiny = [float(i * 0.1) for i in range(1, 13)]
    candidates.append(z_tiny)
    
    # Candidate 17: Exponential growth
    z_exp = [math.exp(i * 0.5) for i in range(1, 13)]
    candidates.append(z_exp)
    
    # Evaluate all candidates
    best_score = float('inf')
    best_z = z_bessel
    
    for z_candidate in candidates:
        if time.time() > deadline - 120:
            break
        try:
            score = _compute_score_fast(z_candidate)
            if score is not None and score < best_score:
                best_score = score
                best_z = list(z_candidate)
        except Exception:
            continue
    
    # Local optimization on the best candidate
    if time.time() < deadline - 60:
        try:
            z_opt, score_opt = _local_optimize(best_z, max_iter=60, step_init=0.05, deadline=deadline)
            if score_opt < best_score:
                best_z = z_opt
                best_score = score_opt
        except Exception:
            pass
    
    # Try local optimization on a few other good candidates
    if time.time() < deadline - 30:
        scored_candidates = []
        for z_candidate in candidates:
            if z_candidate == best_z:
                continue
            score = _compute_score_fast(z_candidate)
            if score is not None:
                scored_candidates.append((score, z_candidate))
        scored_candidates.sort()
        for score, z_candidate in scored_candidates[:5]:
            if time.time() > deadline - 30:
                break
            try:
                z_opt, score_opt = _local_optimize(z_candidate, max_iter=40, step_init=0.05, deadline=deadline)
                if score_opt < best_score:
                    best_z = z_opt
                    best_score = score_opt
            except Exception:
                continue
    
    return best_z

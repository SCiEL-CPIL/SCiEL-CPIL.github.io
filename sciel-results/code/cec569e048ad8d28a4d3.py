"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _laguerre_gen_explicit(n, alpha, x):
    """Compute generalized Laguerre polynomial L_n^(alpha)(x) using explicit formula."""
    # L_n^(alpha)(x) = sum_{k=0}^{n} binom(n+alpha, n-k) * (-1)^k * x^k / k!
    from math import factorial
    result = 0.0
    for k in range(n + 1):
        # Compute binom(n+alpha, n-k) using gamma function
        from math import lgamma
        log_binom = lgamma(n + alpha + 1) - lgamma(n - k + 1) - lgamma(k + 1)
        binom_val = math.exp(log_binom) if log_binom < 700 else 0.0
        coeff = binom_val * ((-1)**k) / factorial(k)
        result += coeff * (x**k)
    return result

def _compute_score_numerical(z_list, max_time=10.0):
    """
    Fast numerical approximation of B/(2*pi).
    Uses floating-point Laguerre polynomials and numerical root finding.
    """
    start_time = time.time()
    z_sorted = sorted(z_list)
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    # We'll evaluate them as functions
    alpha = -0.5
    
    # Precompute coefficient arrays for each Laguerre polynomial
    # L_d^(alpha)(x) = sum_{k=0}^{d} c_k x^k
    lag_coeffs = []
    for d in range(0, 51, 2):
        coeffs = np.zeros(d + 1)
        for k in range(d + 1):
            from math import lgamma, factorial
            log_binom = lgamma(d + alpha + 1) - lgamma(d - k + 1) - lgamma(k + 1)
            binom_val = math.exp(log_binom) if log_binom < 700 else 0.0
            coeffs[k] = binom_val * ((-1)**k) / factorial(k)
        lag_coeffs.append(coeffs)
    
    # Build matrix A and vector b
    # A[i,j] = L_2j^(alpha)(z_i) or derivative
    n_z = 12
    n_pols = 26
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Evaluate Laguerre polynomials and their derivatives at points
    def eval_lag_and_deriv(coeffs, x_val):
        """Evaluate polynomial and its derivative at x_val."""
        # Polynomial: sum coeffs[k] * x^k
        val = 0.0
        deriv = 0.0
        for k in range(len(coeffs)):
            val += coeffs[k] * (x_val**k)
            if k > 0:
                deriv += coeffs[k] * k * (x_val**(k-1))
        return val, deriv
    
    # Row 0: g(0) = 0
    for j in range(26):
        A[0, j] = eval_lag_and_deriv(lag_coeffs[j], 0.0)[0]
    b[0] = 0.0
    
    # Row 1: g'(0) = 1
    for j in range(26):
        A[1, j] = eval_lag_and_deriv(lag_coeffs[j], 0.0)[1]
    b[1] = 1.0
    
    # Rows 2..25
    for i in range(12):
        zi = z_sorted[i]
        row_g = 2 + 2*i
        row_gp = 3 + 2*i
        for j in range(26):
            val, deriv = eval_lag_and_deriv(lag_coeffs[j], zi)
            A[row_g, j] = val
            A[row_gp, j] = deriv
        b[row_g] = 0.0
        b[row_gp] = 0.0
    
    # Solve the linear system
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    # Build g(x) as a polynomial: g(x) = sum_j c[j] * L_2j(x)
    # Convert to standard polynomial form
    g_coeffs = np.zeros(51)
    for j in range(26):
        for k in range(len(lag_coeffs[j])):
            g_coeffs[k] += c[j] * lag_coeffs[j][k]
    
    # Build denominator: x * prod_i (x - z_i)^2
    # This is a polynomial of degree 1 + 24 = 25
    # Start with [0, 1] for x
    denom_coeffs = np.array([0.0, 1.0])
    for zi in z_sorted:
        # Multiply by (x - zi)^2 = x^2 - 2*zi*x + zi^2
        factor = np.array([zi*zi, -2.0*zi, 1.0])
        denom_coeffs = np.convolve(denom_coeffs, factor)
    
    # Polynomial division: g / denom
    # g has degree up to 50, denom has degree 25
    # q should have degree up to 25
    g_poly = g_coeffs.copy()
    denom_poly = denom_coeffs.copy()
    
    # Make sure we have the right length
    deg_g = len(g_poly) - 1
    deg_d = len(denom_poly) - 1
    
    if deg_g < deg_d:
        return 0.0  # q = 0
    
    q_coeffs = np.zeros(deg_g - deg_d + 1)
    remainder = g_poly.copy()
    
    # Long division
    while len(remainder) > deg_d and np.max(np.abs(remainder)) > 1e-15:
        # Find the leading term
        idx = len(remainder) - 1 - deg_d
        if idx < 0:
            break
        lead_coeff = remainder[-1]
        if abs(lead_coeff) < 1e-15:
            break
        q_coeffs[idx] = lead_coeff / denom_poly[-1]
        # Subtract q_coeffs[idx] * x^idx * denom_poly
        shift = len(remainder) - 1 - (deg_d + idx)
        if shift >= 0:
            for k in range(len(denom_poly)):
                pos = len(remainder) - 1 - (deg_d - k) - idx
                if 0 <= pos < len(remainder):
                    remainder[pos] -= q_coeffs[idx] * denom_poly[k]
        # Trim leading zeros
        while len(remainder) > 0 and abs(remainder[-1]) < 1e-15:
            remainder = remainder[:-1]
    
    if len(remainder) > deg_d and np.max(np.abs(remainder)) > 1e-10:
        # Division didn't work cleanly, but we proceed with q_coeffs
        pass
    
    # Find real roots of q using numpy
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return None
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(float(r.real))
    
    if not real_roots:
        return 0.0
    
    # Evaluate q at each real root to check sign change
    def eval_q(x_val):
        """Evaluate q polynomial at x_val."""
        val = 0.0
        for k in range(len(q_coeffs)):
            val += q_coeffs[k] * (x_val**k)
        return val
    
    max_r = 0.0
    eps = 1e-12
    for r in real_roots:
        if r <= 0:
            continue
        q_minus = eval_q(r - eps)
        q_plus = eval_q(r + eps)
        if q_minus * q_plus < 0:
            if r > max_r:
                max_r = r
    
    B = max(0.0, max_r)
    score = B / (2.0 * math.pi)
    return score

def _refine_z_numerical(z_initial, max_iter=50, tol=1e-8):
    """
    Refine z values to minimize B using numerical gradient-free optimization.
    Uses coordinate descent with local perturbations.
    """
    z = list(z_initial)
    best_score = _compute_score_numerical(z)
    if best_score is None:
        return z
    
    for iteration in range(max_iter):
        improved = False
        for i in range(12):
            # Try small perturbations of z[i]
            base = z[i]
            for delta in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]:
                z_new = list(z)
                z_new[i] = base + delta
                if any(zv <= 0 or not math.isfinite(zv) for zv in z_new):
                    continue
                # Check distinctness
                valid = True
                for a in range(12):
                    for b in range(a+1, 12):
                        if abs(z_new[a] - z_new[b]) < 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                score = _compute_score_numerical(z_new)
                if score is not None and score < best_score - 1e-10:
                    z = z_new
                    best_score = score
                    improved = True
                    break
        
        if not improved:
            break
    
    return z

def _time_remaining(deadline):
    return deadline - time.time()

def _validate_z(z_list):
    """Check that z_list is valid: 12 positive, finite, pairwise distinct floats."""
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

def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 580.0  # 630s kill, keep 50s margin
    
    # Initial candidate: geometric progression
    # Start with a reasonable spread
    z_base = [1.0 * (1.3**i) for i in range(12)]
    
    if not _validate_z(z_base):
        return [4.0 * (i + 1) for i in range(12)]
    
    # First, do numerical refinement
    try:
        z_refined = _refine_z_numerical(z_base, max_iter=30, tol=1e-8)
        if _validate_z(z_refined):
            z_base = z_refined
    except Exception:
        pass
    
    # Now try a few different initial configurations
    candidates = [z_base]
    
    # Try Bessel J0 zeros squared
    def bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    if _validate_z(z_bessel):
        candidates.append(z_bessel)
    
    # Try evenly spaced
    z_even = [1.0 + i * 2.0 for i in range(12)]
    if _validate_z(z_even):
        candidates.append(z_even)
    
    # Try to refine each candidate and pick the best
    best_z = z_base
    best_score = float('inf')
    
    for cand in candidates:
        if _time_remaining(deadline) < 60:
            break
        
        try:
            z_ref = _refine_z_numerical(cand, max_iter=20, tol=1e-8)
            if _validate_z(z_ref):
                score = _compute_score_numerical(z_ref)
                if score is not None and score < best_score:
                    best_score = score
                    best_z = z_ref
        except Exception:
            continue
    
    # Final validation
    if not _validate_z(best_z):
        return [4.0 * (i + 1) for i in range(12)]
    
    return best_z
# EVOLVE-BLOCK-END

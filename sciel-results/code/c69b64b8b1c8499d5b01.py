import time
import math
import numpy as np
import scipy.special
import scipy.linalg

def _compute_score_fast(z_list):
    """
    Compute B/(2*pi) for given z_list using numerical methods.
    Returns None if invalid or computation fails.
    """
    z_sorted = sorted([float(z) for z in z_list])
    if len(z_sorted) != 12:
        return None
    if any(z <= 0 for z in z_sorted):
        return None
    # Check distinctness
    for i in range(11):
        if z_sorted[i+1] - z_sorted[i] < 1e-12:
            return None

    # Build Laguerre polynomials L_d^{(-1/2)}(x) for d = 0, 2, 4, ..., 50
    # We need the values and derivatives at the points 0 and z_i.
    # We can use scipy.special.eval_genlaguerre for the values.
    # For derivatives, we use the identity: d/dx L_n^alpha(x) = -L_{n-1}^{alpha+1}(x)
    
    def L_val(d, x_val):
        return scipy.special.eval_genlaguerre(d, -0.5, x_val)
    
    def L_der(d, x_val):
        if d == 0:
            return 0.0
        return -scipy.special.eval_genlaguerre(d-1, 0.5, x_val)

    # Points of evaluation: 0, z_1, ..., z_12
    # We need g(p) and g'(p) for p in {0, z_i}
    # g(x) = sum_{k=0}^{25} c_k L_{2k}^{(-1/2)}(x)
    
    # Matrix A (26x26) and vector b (26)
    # Rows 0, 1 correspond to x=0
    # Rows 2+2i, 3+2i correspond to x=z_i
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Row 0: g(0) = 0
    for k in range(26):
        d = 2*k
        A[0, k] = L_val(d, 0.0)
    b[0] = 0.0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        d = 2*k
        A[1, k] = L_der(d, 0.0)
    b[1] = 1.0
    
    # Rows for z_i
    for i in range(12):
        zi = z_sorted[i]
        row_val = 2 + 2*i
        row_der = 3 + 2*i
        for k in range(26):
            d = 2*k
            A[row_val, k] = L_val(d, zi)
            A[row_der, k] = L_der(d, zi)
        b[row_val] = 0.0
        b[row_der] = 0.0
    
    # Solve for coefficients c
    try:
        c = scipy.linalg.solve(A, b, assume_a='gen')
    except Exception:
        return None

    # Now we have c. We need to construct g(x) as a standard polynomial to divide by denom.
    # g(x) = sum c_k L_{2k}^{(-1/2)}(x)
    # We need the coefficients of L_{2k}^{(-1/2)}(x) in the power basis.
    # L_n^alpha(x) = sum_{k=0}^n (-1)^k binom(n+alpha, n-k) x^k / k!
    
    def laguerre_power_coeffs(n, alpha):
        """Returns list of coefficients [c_0, ..., c_n] for L_n^alpha(x) = sum c_j x^j"""
        coeffs = np.zeros(n+1)
        for k in range(n+1):
            # term is (-1)^k * binom(n+alpha, n-k) / k! * x^k
            # We need binom(n+alpha, n-k).
            # n+alpha is not an integer.
            # binom(z, m) = Gamma(z+1) / (Gamma(m+1) Gamma(z-m+1))
            # Here z = n+alpha, m = n-k
            z = n + alpha
            m = n - k
            if m < 0:
                continue
            # Calculate binomial coefficient using log-gamma for stability if needed, 
            # but for small n (<=50) direct factorial/gamma is fine.
            # scipy.special.gammaln is good for this.
            log_binom = scipy.special.gammaln(z + 1) - scipy.special.gammaln(m + 1) - scipy.special.gammaln(z - m + 1)
            binom_val = math.exp(log_binom)
            coeff = ((-1)**k) * binom_val / math.factorial(k)
            coeffs[k] = coeff
        return coeffs

    # Build g_poly coefficients
    # Max degree is 50.
    g_poly = np.zeros(51)
    for k in range(26):
        d = 2*k
        lc = laguerre_power_coeffs(d, -0.5)
        g_poly[:d+1] += c[k] * lc
    
    # Build denominator: x * prod_{i=1}^{12} (x - z_i)^2
    # Degree is 1 + 24 = 25.
    # We can build this by multiplying polynomials.
    # Start with x
    denom = np.array([0.0, 1.0]) # x
    
    for zi in z_sorted:
        # (x - zi)^2 = x^2 - 2*zi*x + zi^2
        sq = np.array([zi*zi, -2*zi, 1.0])
        # Convolve
        new_denom = np.convolve(denom, sq)
        denom = new_denom
    
    # Polynomial division: g_poly / denom
    # g_poly has degree 50. denom has degree 25.
    # Quotient q will have degree 25.
    
    def poly_div(dividend, divisor):
        """
        dividend, divisor: numpy arrays of coefficients, lowest degree first.
        Returns quotient, remainder.
        """
        # Trim trailing zeros
        def trim(p):
            p = p.copy()
            while len(p) > 0 and abs(p[-1]) < 1e-15:
                p = p[:-1]
            return p
        
        dividend = trim(dividend)
        divisor = trim(divisor)
        
        if len(dividend) < len(divisor):
            return np.array([0.0]), dividend
        
        quotient = np.zeros(len(dividend) - len(divisor) + 1)
        remainder = dividend.copy()
        
        for i in range(len(quotient) - 1, -1, -1):
            # Leading term of remainder is remainder[i + len(divisor) - 1] * x^{i + len(divisor) - 1}
            # Leading term of divisor is divisor[-1] * x^{len(divisor)-1}
            factor = remainder[i + len(divisor) - 1] / divisor[-1]
            quotient[i] = factor
            # Subtract factor * x^i * divisor from remainder
            # divisor has length L. factor * x^i * divisor affects indices i ... i+L-1
            remainder[i : i + len(divisor)] -= factor * divisor
            
        remainder = trim(remainder)
        return quotient, remainder

    try:
        q_poly, rem = poly_div(g_poly, denom)
    except Exception:
        return None
    
    # Find real roots of q_poly
    # q_poly is degree 25.
    # np.roots expects highest degree first.
    q_poly_high = q_poly[::-1]
    
    # Trim leading zeros just in case
    while len(q_poly_high) > 0 and abs(q_poly_high[0]) < 1e-15:
        q_poly_high = q_poly_high[1:]
        
    if len(q_poly_high) == 0:
        return 0.0

    try:
        roots = np.roots(q_poly_high)
    except Exception:
        return None
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    if not real_roots:
        return 0.0
        
    # Define q(x) function for evaluation
    # q_poly is low-to-high
    def q_func(x_val):
        # Use Horner's method for stability
        # Or just sum
        result = 0.0
        # q_poly is [c0, c1, ...]
        # result = c0 + c1*x + ...
        # Horner: result = c_n; result = result*x + c_{n-1} ...
        for coeff in reversed(q_poly):
            result = result * x_val + coeff
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

def construct():
    deadline = time.time() + 570.0
    
    # Generate candidates
    candidates = []
    
    # 1. Bessel zeros squared (known good baseline)
    try:
        bessel_zeros = scipy.special.jn_zeros(0, 12)
        z_bessel = [float(z**2) for z in bessel_zeros]
        candidates.append(z_bessel)
    except Exception:
        # Fallback approximation
        def _bessel_zero_J0_approx(k):
            x = (k - 0.25) * math.pi
            if x < 0.1:
                return 2.4048255576957728
            return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
        z_bessel = [_bessel_zero_J0_approx(i)**2 for i in range(1, 13)]
        candidates.append(z_bessel)
        
    # 2. Laguerre roots (nodes of L_12^(-1/2))
    try:
        roots_lag = scipy.special.roots_genlaguerre(12, -0.5)[0]
        z_lag = sorted([float(r) for r in roots_lag if r > 0])
        if len(z_lag) == 12:
            candidates.append(z_lag)
    except Exception:
        pass
        
    # 3. Geometric progressions
    for ratio in [1.2, 1.5, 2.0, 3.0]:
        z_geo = [1.0 * (ratio ** i) for i in range(12)]
        candidates.append(z_geo)
        
    # 4. Linear progression (integers)
    z_lin = [float(i) for i in range(1, 13)]
    candidates.append(z_lin)
    
    # 5. Quadratic progression
    z_quad = [float(i*i) for i in range(1, 13)]
    candidates.append(z_quad)
    
    best_score = float('inf')
    best_z = candidates[0] # Default to Bessel
    
    # Evaluate candidates
    # We evaluate in order of likely performance or just iterate.
    # Bessel is usually strong.
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
            
    # If we have time left, we could try to optimize the best candidate slightly,
    # but the task asks to keep the design. The current design is "evaluate a set of candidates".
    # The "Spectral Interpolation" inspiration suggests that Laguerre nodes are good.
    # The "Current Program" already includes them.
    # The main improvement from inspirations is ensuring robustness and maybe adding a few more structured candidates.
    
    # Let's add a few more structured candidates if time permits, specifically variations of Laguerre nodes
    # or nodes from other orthogonal polynomials if easily available.
    # But for now, the existing set covers the main bases.
    
    return best_z

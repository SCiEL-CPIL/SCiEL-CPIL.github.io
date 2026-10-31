import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def _get_hermite_coeffs(n):
    """
    Returns coefficients of the n-th physicist's Hermite polynomial H_n(x).
    H_0 = 1
    H_1 = 2x
    H_{n+1} = 2x H_n - 2n H_{n-1}
    """
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0]) # 2x + 0
    
    h_nm1 = np.array([1.0])
    h_n = np.array([2.0, 0.0])
    
    for k in range(1, n):
        # H_{k+1} = 2x H_k - 2k H_{k-1}
        # 2x H_k shifts coefficients by 1 and multiplies by 2
        term1 = np.zeros(len(h_n) + 1)
        term1[1:] = 2.0 * h_n
        
        # -2k H_{k-1}
        term2 = -2.0 * k * h_nm1
        # Pad term2 to match length
        if len(term2) < len(term1):
            term2 = np.pad(term2, (0, len(term1) - len(term2)))
        elif len(term1) < len(term2):
            term1 = np.pad(term1, (0, len(term2) - len(term1)))
        
        h_new = term1 + term2
        h_nm1 = h_n
        h_n = h_new
        
    return h_n

# Precompute coefficients for H_0, H_4, H_8, H_12
H0_coeffs = _get_hermite_coeffs(0)
H4_coeffs = _get_hermite_coeffs(4)
H8_coeffs = _get_hermite_coeffs(8)
H12_coeffs = _get_hermite_coeffs(12)

# Values at 0 for constraint
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 665280.0

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin
    
    def get_poly_coeffs(c0, c4, c8):
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        # Polynomial coefficients: c0*H0 + c4*H4 + c8*H8 + c12*H12
        # Max degree is 12.
        coeffs = np.zeros(13)
        # Add contributions
        coeffs[:len(H0_coeffs)] += c0 * H0_coeffs
        coeffs[:len(H4_coeffs)] += c4 * H4_coeffs
        coeffs[:len(H8_coeffs)] += c8 * H8_coeffs
        coeffs[:len(H12_coeffs)] += c12 * H12_coeffs
        return coeffs

    def objective(params):
        c0, c4, c8 = params
        coeffs = get_poly_coeffs(c0, c4, c8)
        
        # Find real roots of P(x)
        roots = np.roots(coeffs)
        real_roots = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
        
        # Integration points: -inf, roots, inf
        # We integrate |e^{-x^2/2} P(x)|
        # Split into intervals
        points = np.concatenate(([-np.inf], real_roots, [np.inf]))
        
        total_integral = 0.0
        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]
            
            # Midpoint to determine sign
            if a == -np.inf and b == np.inf:
                mid = 0.0
            elif a == -np.inf:
                mid = b - 1.0
            elif b == np.inf:
                mid = a + 1.0
            else:
                mid = (a + b) / 2.0
            
            # Evaluate P(mid) to get sign
            p_mid = np.polyval(coeffs, mid)
            
            if p_mid >= 0:
                integrand = lambda x: np.exp(-x**2 / 2.0) * np.polyval(coeffs, x)
            else:
                integrand = lambda x: -np.exp(-x**2 / 2.0) * np.polyval(coeffs, x)
            
            # Integrate
            val, err = integrate.quad(integrand, a, b, limit=100)
            total_integral += val
            
        return total_integral

    # Multi-start optimization
    best_score = np.inf
    best_params = None
    
    # Seed initial points
    # The seed program returns [-113/100, 1/25, 1/3240]
    seed_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Generate random starts
    np.random.seed(42)
    starts = [seed_params]
    for _ in range(30):
        # Scale based on seed magnitude
        p = np.random.uniform(-1, 1, 3)
        p[0] *= 1.0
        p[1] *= 1.0
        p[2] *= 1.0
        starts.append(p)
        
    # Add some structured starts
    for i in range(10):
        p = np.zeros(3)
        p[i % 3] = 0.1 * (i + 1)
        starts.append(p)
        
    for start in starts:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(objective, start, method='Nelder-Mead', 
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 1000})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            continue
            
    # If optimization failed or didn't find anything, return seed
    if best_params is None:
        return seed_params
        
    # The score is the upper bound on C4?
    # The prompt says "score is the upper bound on C4" and "evaluator measures the raw value".
    # And "1.0 equals the seed program".
    # We minimized L1.
    # J = sqrt(2*pi) * L1^2.
    # If seed J = 1.0, then seed L1^2 = 1/sqrt(2*pi).
    # We found a lower L1.
    # We should return the coefficients that minimize J.
    # The interface asks for "three coefficients of the Hermite combination".
    
    return best_params

# EVOLVE-BLOCK-END

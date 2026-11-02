"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Coefficients of Physicist's Hermite Polynomials H_k(x) in power basis (highest degree first)
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H_12(x) = 4096x^12 - 46080x^10 + 241920x^8 - 580608x^6 + 645120x^4 - 302400x^2 + 4608

# We need coefficients for g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# Let's define the coefficient vectors for each H_k, padded to length 13 (degree 12)
# Order: [x^12, x^11, ..., x^0]

H0_coeff = np.zeros(13)
H0_coeff[-1] = 1.0

H4_coeff = np.zeros(13)
H4_coeff[-1] = 12.0
H4_coeff[-3] = -48.0
H4_coeff[-5] = 16.0

H8_coeff = np.zeros(13)
H8_coeff[-1] = 168.0
H8_coeff[-3] = -1344.0
H8_coeff[-5] = 2688.0
H8_coeff[-7] = -1344.0
H8_coeff[-9] = 256.0

H12_coeff = np.zeros(13)
H12_coeff[-1] = 4608.0
H12_coeff[-3] = -302400.0
H12_coeff[-5] = 645120.0
H12_coeff[-7] = -580608.0
H12_coeff[-9] = 241920.0
H12_coeff[-11] = -46080.0
H12_coeff[-13] = 4096.0

def _compute_upper_bound_exact(coeffs):
    """
    Compute the upper bound A(f)*A(F) using exact root finding for extrema.
    coeffs = [c0, c4, c8]
    """
    c0, c4, c8 = coeffs
    
    # c12 is determined by g(0) = 0
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)
    # H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=4608
    c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 168.0) / 4608.0
    
    # Construct g(x) coefficients
    g_coeff = c0 * H0_coeff + c4 * H4_coeff + c8 * H8_coeff + c12 * H12_coeff
    
    # We want to maximize |f(x)| = |g(x) exp(-x^2/2)|
    # Extrema occur where f'(x) = 0 => g'(x) - x*g(x) = 0
    # Let h(x) = g'(x) - x*g(x)
    # g(x) = sum_{j=0}^{12} G_{12-j} x^j  (using numpy order: coeff[0] is x^12)
    # Actually, let's stick to numpy.roots convention: coeff[0] * x^N + ... + coeff[N]
    
    # g_coeff is in order [x^12, ..., x^0]
    # g'(x) coefficients:
    # If g(x) = c_12 x^12 + ... + c_0
    # g'(x) = 12 c_12 x^11 + ... + c_1
    # x*g(x) = c_12 x^13 + ... + c_0 x
    
    # Let's form h(x) = g'(x) - x*g(x)
    # Degree of x*g(x) is 13. Degree of g'(x) is 11.
    # So h(x) is degree 13.
    
    # Coefficients of x*g(x) in order [x^13, ..., x^0]:
    # x*g(x) = c_12 x^13 + c_11 x^12 + ... + c_0 x
    # coeff_xg[0] = g_coeff[0] (c_12)
    # coeff_xg[1] = g_coeff[1] (c_11)
    # ...
    # coeff_xg[13] = 0 (constant term)
    coeff_xg = np.zeros(14)
    coeff_xg[0:13] = g_coeff[0:13]
    
    # Coefficients of g'(x) in order [x^13, ..., x^0]:
    # g'(x) has max degree 11.
    # coeff_gprime[0:2] = 0 (x^13, x^12 terms)
    # coeff_gprime[2] = 12 * g_coeff[0]
    # coeff_gprime[3] = 11 * g_coeff[1]
    # ...
    # coeff_gprime[k] = (12 - (k-2)) * g_coeff[k-2] = (14-k) * g_coeff[k-2] for k=2..12
    # coeff_gprime[13] = g_coeff[12] (derivative of constant term c_0 is c_0? No.
    # g(x) = ... + c_1 x + c_0. g'(x) = ... + c_1.
    # So coeff_gprime[13] corresponds to x^0 term in derivative?
    # Let's index carefully.
    # g_coeff[i] is coeff of x^(12-i).
    # g'(x) coeff of x^(11-i) is (12-i) * g_coeff[i].
    # We want to align to 14-length vector (x^13 to x^0).
    # Term x^k in g'(x) comes from term x^(k+1) in g(x).
    # Coeff of x^k in g' is (k+1) * (coeff of x^(k+1) in g).
    # Let j be index in g_coeff (0 to 12). Power is 12-j.
    # Derivative power is 11-j.
    # In 14-length vector (index 0 is x^13, index 13 is x^0), the index for power p is 13-p.
    # So index for derivative term is 13 - (11-j) = 2+j.
    # Value is (12-j) * g_coeff[j].
    
    coeff_gprime = np.zeros(14)
    for j in range(13):
        if 12 - j > 0: # Derivative of constant is 0, but formula gives 0 * c_0 = 0?
            # Wait. If j=12, power is 0. Derivative is 0.
            # Formula: (12-j) * g_coeff[j]. If j=12, 0 * c_0 = 0. Correct.
            coeff_gprime[2+j] = (12 - j) * g_coeff[j]
    
    # h(x) = g'(x) - x*g(x)
    h_coeff = coeff_gprime - coeff_xg
    
    # Find roots of h(x)
    try:
        roots = np.roots(h_coeff)
    except Exception:
        # Fallback to grid if roots fail
        x = np.linspace(-10, 10, 10001)
        f_vals = np.polynomial.polynomial.polyval(x, g_coeff[::-1]) * np.exp(-x**2 / 2)
        A_f = np.max(np.abs(f_vals))
        return A_f * A_f
    
    # Filter real roots
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-10])
    
    if len(real_roots) == 0:
        # Should not happen for real polynomials of odd degree? h is degree 13 (odd).
        # Must have at least one real root.
        x = np.linspace(-10, 10, 10001)
        f_vals = np.polynomial.polynomial.polyval(x, g_coeff[::-1]) * np.exp(-x**2 / 2)
        A_f = np.max(np.abs(f_vals))
        return A_f * A_f

    # Evaluate f(x) at real roots
    # f(x) = g(x) exp(-x^2/2)
    # Use Horner's method or polyval. g_coeff is [x^12, ..., x^0]
    # np.polyval expects [x^N, ..., x^0]
    g_vals = np.polyval(g_coeff, real_roots)
    f_vals = g_vals * np.exp(-real_roots**2 / 2)
    
    A_f = np.max(np.abs(f_vals))
    
    # Since i^k = 1 for k in {0,4,8,12}, F(f) = f.
    # So A(F) = A(f).
    return A_f * A_f

def construct():
    deadline = time.time() + 170  # Safety margin
    
    # Initial guess from Goncalves et al. 2017
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_score = _compute_upper_bound_exact(x0)
    best_coeffs = x0.copy()
    
    # Optimization
    try:
        # Use Nelder-Mead with tight tolerances
        result = minimize(_compute_upper_bound_exact, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_coeffs
    
    # Random restarts
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Perturb around best solution
        x_try = best_coeffs + np.random.normal(0, 0.05, 3)
        
        try:
            result = minimize(_compute_upper_bound_exact, x_try, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END

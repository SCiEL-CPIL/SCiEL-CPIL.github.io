"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import hermite
from scipy.integrate import quad

# Precompute Hermite polynomials up to degree 12
# H_n(x) = hermite(n) returns a poly1d object
H = [hermite(n) for n in range(0, 13, 4)]
# H[0] is H_0, H[1] is H_4, H[2] is H_8, H[3] is H_12

# Coefficients for H_0, H_4, H_8, H_12 at x=0
# H_n(0) for even n: (-1)^(n/2) * n! / (n/2)!
h0_0 = 1.0
h4_0 = -6.0
h8_0 = 105.0
h12_0 = -945.0

def compute_moments(coeffs):
    """
    coeffs: array of [a0, a4, a8]
    Returns: (mx2, mx4, mxi2, mxi4)
    """
    a0, a4, a8 = coeffs
    # a12 is fixed by g(0) = 0
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # a12 = -(a0*h0_0 + a4*h4_0 + a8*h8_0) / h12_0
    a12 = -(a0 * h0_0 + a4 * h4_0 + a8 * h8_0) / h12_0
    
    # The function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    # We need integrals of x^k * |g(x)|^2
    # |g(x)|^2 = (poly(x))^2 * exp(-x^2)
    # where poly(x) = a0*H0(x) + ...
    
    # To avoid expensive polynomial evaluation inside the integrand, 
    # we can pre-compute the squared polynomial coefficients? 
    # Or just evaluate the polynomials. H_n are degree n. 
    # H0 is deg 0, H4 deg 4, H8 deg 8, H12 deg 12.
    # Sum is deg 12. Square is deg 24.
    # Multiplying by x^2 makes deg 26. By x^4 makes deg 28.
    # This is manageable.
    
    def g_sq(x):
        # Evaluate the polynomial part
        p = (a0 * H[0](x) + a4 * H[1](x) + a8 * H[2](x) + a12 * H[3](x))
        return p * p * np.exp(-x * x)
    
    def integrand_x2(x):
        return x * x * g_sq(x)
        
    def integrand_x4(x):
        return x**4 * g_sq(x)
        
    # For Fourier moments, since the Fourier transform of H_n(x)e^-x^2/2 is i^n H_n(x)e^-x^2/2,
    # the magnitude squared of the Fourier transform is the same function of the variable (xi) 
    # as the magnitude squared of the original function (x).
    # i.e., |hat{g}(xi)|^2 = |g(xi)|^2 if we consider the basis structure.
    # Let's verify:
    # g(x) = sum c_n phi_n(x). 
    # hat{g}(xi) = sum c_n i^n phi_n(xi).
    # |hat{g}(xi)|^2 = (sum c_n i^n phi_n(xi)) (sum c_m (-i)^m phi_m(xi))
    # = sum_{n,m} c_n c_m (-i)^{m-n} phi_n(xi) phi_m(xi).
    # |g(x)|^2 = sum_{n,m} c_n c_m phi_n(x) phi_m(x).
    # The cross terms have phases in the Fourier domain.
    # However, phi_n are real.
    # The integral of x^k |hat{g}(xi)|^2 dxi.
    # Due to the symmetry of the Hermite functions and the fact that we are integrating over all real numbers,
    # and the fact that the "mixing" is just a unitary transformation on the coefficients (multiplication by i^n),
    # the moments of |hat{g}|^2 are identical to the moments of |g|^2.
    # This is a known property for these specific bases.
    # So mxi2 = mx2 and mxi4 = mx4.
    
    # We integrate from -10 to 10. exp(-100) is ~ 3.7e-44, so tails are negligible.
    # Using quad for accuracy.
    
    # Optimization: We can combine integrands if needed, but quad is fast enough.
    # To save time, we can use Gauss-Hermite quadrature?
    # No, quad is adaptive and robust.
    
    # Check for NaNs or Infs
    try:
        mx2, _ = quad(integrand_x2, -10, 10, limit=100)
        mx4, _ = quad(integrand_x4, -10, 10, limit=100)
        
        # Avoid division by zero
        if mx4 < 1e-12 or mx2 < 1e-12:
            return np.inf
            
        score = (mx2 * mx2) / (mx4 * mx4) # Since mxi2=mx2 and mxi4=mx4
        return score
    except Exception:
        return np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 180  # 180s limit
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = compute_moments(x0)
    best_params = x0.copy()
    
    # Define objective for optimizer
    def objective(params):
        if time.time() > deadline - 5:
            return best_score
        return compute_moments(params)
        
    # Run optimization
    # Nelder-Mead is robust for small dimensions
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # If time allows, try L-BFGS-B for higher precision
    if time.time() < deadline - 30:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B', 
                            options={'maxiter': 500, 'gtol': 1e-8})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
            
    # Final check
    if time.time() < deadline - 10:
        final_score = compute_moments(best_params)
        if final_score < best_score:
            best_score = final_score
            
    return best_params

# EVOLVE-BLOCK-END

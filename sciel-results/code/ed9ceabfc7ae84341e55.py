import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # safety margin before 210s kill

    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 166320x^4 - 302400x^2 + 10395

    # Precompute polynomial coefficients for numerical evaluation
    # Using Horner's method for stability
    H0_coeffs = np.array([1.0])
    H4_coeffs = np.array([1.0, 0.0, -6.0, 0.0, 3.0])
    H8_coeffs = np.array([1.0, 0.0, -28.0, 0.0, 210.0, 0.0, -420.0, 0.0, 105.0])
    H12_coeffs = np.array([1.0, 0.0, -84.0, 0.0, 2520.0, 0.0, -30240.0, 0.0, 
                           166320.0, 0.0, -302400.0, 0.0, 10395.0])

    def _eval_poly(coeffs, x):
        """Evaluate polynomial using Horner's method. coeffs is [a_n, a_{n-1}, ..., a_0]."""
        result = np.zeros_like(x, dtype=np.float64)
        for c in coeffs:
            result = result * x + c
        return result

    def _compute_g(c0, c4, c8, x):
        """Compute g(x) = P(x) * exp(-x^2/2) where P is the Hermite combination."""
        c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
        
        H0 = np.ones_like(x)
        H4 = _eval_poly(H4_coeffs, x)
        H8 = _eval_poly(H8_coeffs, x)
        H12 = _eval_poly(H12_coeffs, x)
        
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        return P * np.exp(-x**2 / 2.0)

    def _compute_bound(c0, c4, c8, x_samples):
        """
        Compute the upper bound C4 on A(f)*A(Fourier(f)).
        Using the identity: <k^2> = <(g')^2> / <g^2> for normalized functions.
        Bound = 4 * sigma_x2 * sigma_k2
        """
        g_vals = _compute_g(c0, c4, c8, x_samples)
        
        g2 = g_vals**2
        int_g2 = np.trapz(g2, x_samples)
        if abs(int_g2) < 1e-15:
            return np.inf
            
        int_x2g2 = np.trapz(x_samples**2 * g2, x_samples)
        sigma_x2 = int_x2g2 / int_g2
        
        # Compute derivative g'(x) analytically to improve accuracy
        # g(x) = P(x) * exp(-x^2/2)
        # g'(x) = P'(x) * exp(-x^2/2) + P(x) * (-x) * exp(-x^2/2)
        #       = (P'(x) - x * P(x)) * exp(-x^2/2)
        
        c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
        
        # Derivatives of Hermite polynomials:
        # H0' = 0
        # H4' = 4x^3 - 12x
        # H8' = 8x^7 - 168x^5 + 840x^3 - 840x
        # H12' = 12x^11 - 840x^9 + 20160x^7 - 181440x^5 + 665280x^3 - 604800x
        
        x = x_samples
        H0_prime = np.zeros_like(x)
        H4_prime = 4.0 * x**3 - 12.0 * x
        H8_prime = 8.0 * x**7 - 168.0 * x**5 + 840.0 * x**3 - 840.0 * x
        H12_prime = (12.0 * x**11 - 840.0 * x**9 + 20160.0 * x**7 - 
                     181440.0 * x**5 + 665280.0 * x**3 - 604800.0 * x)
        
        P_prime = c0 * H0_prime + c4 * H4_prime + c8 * H8_prime + c12 * H12_prime
        
        # P(x) again
        H0 = np.ones_like(x)
        H4 = _eval_poly(H4_coeffs, x)
        H8 = _eval_poly(H8_coeffs, x)
        H12 = _eval_poly(H12_coeffs, x)
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        
        g_prime = (P_prime - x * P) * np.exp(-x**2 / 2.0)
        
        int_dg2 = np.trapz(g_prime**2, x_samples)
        sigma_k2 = int_dg2 / int_g2
        
        bound = 4.0 * sigma_x2 * sigma_k2
        return bound

    # Use a finer grid for higher accuracy
    x_samples = np.linspace(-20, 20, 4000)
    
    def objective(vars):
        c0, c4, c8 = vars
        return _compute_bound(c0, c4, c8, x_samples)
    
    def constraint_g_leq_1(vars):
        c0, c4, c8 = vars
        vals = _compute_g(c0, c4, c8, x_samples)
        return 1.0 - np.max(np.abs(vals))
    
    best_bound = np.inf
    best_vars = None
    
    # More diverse starting points
    starts = [
        [1.0, 0.0, 0.0],
        [0.5, 0.1, 0.0],
        [1.0, -0.1, 0.0],
        [0.8, 0.0, 0.1],
        [1.2, 0.0, -0.1],
        [1.5, 0.05, 0.0],
        [0.5, -0.05, 0.02],
        [1.0, 0.05, -0.02],
        [-1.13, 0.04, 1.0/3240.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, -0.2, 0.01],
        [0.3, 0.3, -0.05],
        [-0.5, 0.5, 0.02],
    ]
    
    for x0 in starts:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(objective, x0, method='SLSQP',
                           constraints=[{'type': 'ineq', 'fun': constraint_g_leq_1}],
                           options={'maxiter': 300, 'ftol': 1e-12})
            
            if res.success or res.fun < best_bound:
                bound = objective(res.x)
                if bound < best_bound and constraint_g_leq_1(res.x) >= -1e-6:
                    best_bound = bound
                    best_vars = res.x
        except Exception:
            pass
    
    # Polish with tighter tolerance
    if best_vars is not None and time.time() < deadline - 20:
        try:
            res2 = minimize(objective, best_vars, method='SLSQP',
                            constraints=[{'type': 'ineq', 'fun': constraint_g_leq_1}],
                            options={'maxiter': 500, 'ftol': 1e-14})
            bound2 = objective(res2.x)
            if bound2 < best_bound and constraint_g_leq_1(res2.x) >= -1e-6:
                best_bound = bound2
                best_vars = res2.x
        except Exception:
            pass
    
    if best_vars is None:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    return np.array(best_vars)

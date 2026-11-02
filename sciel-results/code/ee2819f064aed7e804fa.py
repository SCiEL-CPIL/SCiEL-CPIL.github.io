import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    """
    Constructs the coefficients for the Hermite-Gaussian test function
    to minimize the uncertainty constant upper bound.
    Uses exact analytic Fourier transform for precision.
    """
    start_time = time.time()
    max_time = 195.0  # safety margin before 210s kill

    # Hermite polynomials (physicists): H_n(x)
    # Precompute coefficients (index = power of x)
    H0_coeffs = [1.0]
    H4_coeffs = [12.0, 0.0, -48.0, 0.0, 16.0]
    H8_coeffs = [1680.0, 0.0, -13440.0, 0.0, 13440.0, 0.0, -3584.0, 0.0, 256.0]
    H12_coeffs = [10395.0, 0.0, -4636800.0, 0.0, 6220800.0, 0.0, -3184320.0, 0.0, 704448.0, 0.0, -75264.0, 0.0, 4096.0]

    # Precompute all Hermite polynomial coefficients H_0 through H_12
    def hermite_coeffs(n):
        """Return coefficients of H_n(x) (index = power of x)."""
        if n == 0:
            return np.array([1.0])
        if n == 1:
            return np.array([0.0, 2.0])
        h_prev = hermite_coeffs(n-1)
        h_curr = hermite_coeffs(n-2)
        h_next = np.zeros(n+1)
        for i in range(n+1):
            term1 = 0.0
            if i >= 1:
                term1 = 2.0 * h_curr[i-1]
            term2 = -2.0 * (n-1) * h_prev[i] if i < len(h_prev) else 0.0
            h_next[i] = term1 + term2
        return h_next

    H_coeffs_cache = {}
    for k in range(13):
        H_coeffs_cache[k] = hermite_coeffs(k)

    def get_c12(c0, c4, c8):
        return -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0

    def poly_eval(coeffs, x):
        """Evaluate polynomial with coefficients (index = power) at x."""
        result = np.zeros_like(x)
        for i, c in enumerate(coeffs):
            if c != 0:
                result += c * (x ** i)
        return result

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Build combined polynomial P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        P = np.zeros(13)
        for i in range(len(H0_coeffs)):
            P[i] += c0 * H0_coeffs[i]
        for i in range(len(H4_coeffs)):
            P[i] += c4 * H4_coeffs[i]
        for i in range(len(H8_coeffs)):
            P[i] += c8 * H8_coeffs[i]
        for i in range(len(H12_coeffs)):
            P[i] += c12 * H12_coeffs[i]
        
        # L1 norm: ||g||_1 = integral |P(x) e^{-x^2/2}| dx
        # Use adaptive quadrature on a fine grid
        x = np.linspace(-12, 12, 4096)
        g = poly_eval(P, x) * np.exp(-0.5 * x**2)
        l1_norm = np.trapz(np.abs(g), x)
        
        if l1_norm < 1e-15:
            return 1e10
        
        # Build Q(xi) = sum_k P[k] * (-i)^k * H_k(xi)
        # Since P[k] = 0 for odd k, and (-i)^even = (-1)^(k/2),
        # Q(xi) = sum_{k even} P[k] * (-1)^(k/2) * H_k(xi)
        Q = np.zeros(13)
        for k in range(0, 13, 2):
            if abs(P[k]) > 1e-15:
                factor = (-1.0) ** (k // 2)
                Hk = H_coeffs_cache[k]
                for i in range(len(Hk)):
                    Q[i] += P[k] * factor * Hk[i]
        
        # Find min |Q(xi) e^{-xi^2/2}| over xi
        xi = np.linspace(-12, 12, 8192)
        Q_val = poly_eval(Q, xi)
        g_hat_mag = np.abs(Q_val) * np.exp(-0.5 * xi**2)
        
        min_g_hat = np.min(g_hat_mag)
        
        if min_g_hat < 1e-15:
            return 1e10
        
        return (l1_norm ** 2) / min_g_hat

    # Initial guess
    initial_guess = np.array([-1.13, 0.04, 1.0/3240.0])
    
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    
    best_result = None
    best_score = np.inf
    
    elapsed = time.time() - start_time
    if elapsed < max_time:
        try:
            # Global search with differential evolution
            result = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-10,
                polish=False
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
            elapsed = time.time() - start_time
            if elapsed < max_time - 20:
                # Local refinement with multiple methods
                res_local = minimize(
                    objective, 
                    best_result, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15}
                )
                
                if res_local.fun < best_score:
                    best_score = res_local.fun
                    best_result = res_local.x
                    
                # Additional refinement with Nelder-Mead
                elapsed = time.time() - start_time
                if elapsed < max_time - 10:
                    res_nm = minimize(
                        objective, 
                        best_result, 
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    
                    if res_nm.fun < best_score:
                        best_score = res_nm.fun
                        best_result = res_nm.x
                        
        except Exception:
            pass

    if best_result is None:
        return initial_guess
    
    return np.array(best_result)

import numpy as np
from scipy.special import jn_zeros, eval_genlaguerre
from scipy.optimize import minimize
import time

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 580.0  # Evaluator limit, use 580 to be safe
    
    n_roots = 12
    alpha = -0.5
    
    # Precompute coefficients for Laguerre polynomials L_{2k}^{alpha}(x) in monomial basis
    # L_n^alpha(x) = sum_{j=0}^n (-1)^j binom(n+alpha, n-j) x^j / j!
    lag_coeffs = []
    for k in range(26):
        n = 2 * k
        coeffs = np.zeros(n + 1)
        for j in range(n + 1):
            # binom(n+alpha, n-j)
            # Using gamma function for general alpha
            # binom(n+alpha, n-j) = Gamma(n+alpha+1) / (Gamma(n-j+1) * Gamma(n-(n-j)+alpha+1))
            # = Gamma(n+alpha+1) / (Gamma(n-j+1) * Gamma(j+alpha+1))
            # We can compute this recursively or using math.lgamma
            # Let's use a stable calculation.
            # For small n, direct product is fine.
            # binom(n+alpha, n-j)
            # Let's use scipy.special.comb? No, alpha is float.
            # We can use:
            # c = 1.0
            # for m in range(n-j+1):
            #    c *= (n+alpha - (n-j-m)) / (m+1)  # This is binom(n+alpha, n-j)
            # Actually, it's easier to just evaluate the polynomial values and interpolate? No, we need coeffs.
            
            # Let's use the formula:
            # L_n^alpha(x) = sum_{k=0}^n (-1)^k binom(n+alpha, n-k) x^k / k!
            # Here j is the power of x.
            # coeff_j = (-1)^j * binom(n+alpha, n-j) / j!
            
            # Calculate binom(n+alpha, n-j)
            # If n-j is large, this might be tricky.
            # n <= 50.
            # Let's use math.lgamma
            import math
            try:
                log_num = math.lgamma(n + alpha + 1)
                log_den1 = math.lgamma(n - j + 1)
                log_den2 = math.lgamma(j + alpha + 1)
                log_binom = log_num - log_den1 - log_den2
                binom_val = np.exp(log_binom)
            except OverflowError:
                binom_val = 0.0
            
            coeffs[j] = ((-1)**j) * binom_val / math.factorial(j)
        lag_coeffs.append(coeffs)
    
    # Pad to max degree 50
    max_deg = 50
    lag_coeffs_padded = [np.pad(c, (0, max_deg - len(c) + 1)) for c in lag_coeffs]
    
    def compute_B_fast(z_arr):
        try:
            # Build matrix A and vector b for c_k
            # g(x) = sum c_k L_{2k}(x)
            # g(0) = 0 => sum c_k L_{2k}(0) = 0
            # g'(0) = 1 => sum c_k L'_{2k}(0) = 1
            # g(z_i) = 0, g'(z_i) = 0
            
            A = np.zeros((26, 26))
            b = np.zeros(26)
            
            # Row 0: g(0) = 0
            for k in range(26):
                n = 2*k
                A[0, k] = eval_genlaguerre(n, alpha, 0.0)
            
            # Row 1: g'(0) = 1
            for k in range(26):
                n = 2*k
                if n == 0:
                    deriv = 0.0
                else:
                    # L_n^alpha'(x) = -L_{n-1}^{alpha+1}(x)
                    deriv = -eval_genlaguerre(n-1, alpha+1, 0.0)
                A[1, k] = deriv
            b[1] = 1.0
            
            for i in range(12):
                zi = z_arr[i]
                for k in range(26):
                    n = 2*k
                    A[2+i, k] = eval_genlaguerre(n, alpha, zi)
                    if n > 0:
                        A[14+i, k] = -eval_genlaguerre(n-1, alpha+1, zi)
                    else:
                        A[14+i, k] = 0.0
            
            # Solve for c
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return 1e100
            
            # Construct g(x) in monomial basis
            # g(x) = sum c_k L_{2k}(x)
            g_mono = np.zeros(max_deg + 1)
            for k in range(26):
                g_mono += c[k] * lag_coeffs_padded[k]
            
            # Construct D(x) = x * prod (x - z_i)^2
            # D(x) is degree 25
            # Start with x
            D = np.array([1.0, 0.0]) # x
            for i in range(12):
                # Multiply by (x - z_i)^2 = x^2 - 2 z_i x + z_i^2
                term = np.array([1.0, -2.0*z_arr[i], z_arr[i]**2])
                D = np.convolve(D, term)
            
            # Divide g_mono by D to get q(x)
            # g_mono has degree 50, D has degree 25.
            # q(x) should have degree 25.
            q_coeffs, remainder = np.polydiv(g_mono, D)
            
            # Check if remainder is small (sanity check)
            # If not, the float solution might be noisy, but we proceed.
            
            # Find roots of q(x)
            roots = np.roots(q_coeffs)
            real_roots = roots[np.isreal(roots)].real
            if len(real_roots) == 0:
                return 1e100
            
            B = np.max(real_roots)
            return B
            
        except Exception:
            return 1e100

    # Initial guess: Bessel zeros
    try:
        z_bessel = jn_zeros(0, n_roots)
        z0 = np.array(z_bessel, dtype=float)
    except Exception:
        z0 = np.linspace(2, 40, 12)

    # Optimize z
    # We want to minimize B
    # Use Nelder-Mead
    # Constraints: z_i > 0, distinct.
    # We can transform variables to ensure positivity, but Nelder-Mead doesn't support bounds well.
    # We can just penalize invalid z in the objective function.
    
    def objective(z):
        if np.any(z <= 0) or np.any(np.diff(np.sort(z)) < 1e-6):
            return 1e100
        return compute_B_fast(z)
    
    # Run optimization
    # Limit time
    t_start = time.time()
    best_z = z0.copy()
    best_val = objective(best_z)
    
    # Try a few optimization steps
    # Nelder-Mead is derivative-free, good for this.
    # maxiter should be large enough but we have time limits.
    # Each eval is ~ms.
    
    try:
        res = minimize(objective, z0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_z = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return list(best_z)
    
    # Refine with tighter tolerance if time permits
    try:
        res2 = minimize(objective, best_z, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res2.fun < best_val:
            best_z = res2.x
            best_val = res2.fun
    except Exception:
        pass

    # Ensure sorted and positive
    best_z = np.sort(best_z)
    if np.any(best_z <= 0):
        return list(z0)
        
    return list(best_z)

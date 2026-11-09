"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import scipy.integrate as integrate
import time

# Pre-compute power basis coefficients for Hermite polynomials H0, H4, H8, H12
# Only even powers are present.
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 98304x^10 + 1182720x^8 - 6635520x^6 + 15482880x^4 - 1451520x^2 + 665280

# Coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
H0_pow = np.array([1, 0, 0, 0, 0, 0, 0])
H4_pow = np.array([12, -48, 16, 0, 0, 0, 0])
H8_pow = np.array([1680, -13440, 11520, -3072, 256, 0, 0])
H12_pow = np.array([665280, -1451520, 15482880, -6635520, 1182720, -98304, 4096])

# Values at 0 for constraint P(0)=0
H0_0 = H0_pow[0]
H4_0 = H4_pow[0]
H8_0 = H8_pow[0]
H12_0 = H12_pow[0]

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin before 210s kill

    def calculate_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def eval_P(c0, c4, c8, x):
        """Evaluate P(x) using power basis for speed."""
        c12 = calculate_c12(c0, c4, c8)
        # Precompute total coefficients once per coefficient set
        coeff_total = c0 * H0_pow + c4 * H4_pow + c8 * H8_pow + c12 * H12_pow
        # Horner-like evaluation for better numerical stability and speed
        x2 = x * x
        # Evaluate polynomial in x^2 using Horner's method
        result = coeff_total[6]
        for i in range(5, -1, -1):
            result = result * x2 + coeff_total[i]
        return result

    def exact_objective(coeffs):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        
        def integrand(x):
            P = eval_P(c0, c4, c8, x)
            return np.abs(P) * np.exp(-x**2 / 4.0)
            
        try:
            result, error = integrate.quad(
                integrand, -15, 15, limit=200, epsabs=1e-12, epsrel=1e-12
            )
            return result
        except Exception:
            return np.inf

    def smooth_objective(coeffs, eps=1e-4):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        coeff_total = c0 * H0_pow + c4 * H4_pow + c8 * H8_pow + c12 * H12_pow
        
        def integrand(x):
            x2 = x * x
            # Horner-like evaluation
            result = coeff_total[6]
            for i in range(5, -1, -1):
                result = result * x2 + coeff_total[i]
            # Smooth approximation of |P|
            return np.sqrt(result*result + eps) * np.exp(-x2 / 4.0)
            
        try:
            result, error = integrate.quad(
                integrand, -15, 15, limit=200, epsabs=1e-10, epsrel=1e-10
            )
            return result
        except Exception:
            return np.inf

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = x0.copy()
    best_val = exact_objective(x0)
    
    # Quick local refinement from seed to get a better starting point
    if time.time() < deadline - 100:
        try:
            res_quick = opt.minimize(
                lambda c: smooth_objective(c, 1e-4), 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_quick.fun < best_val:
                exact_val = exact_objective(res_quick.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_result = res_quick.x.copy()
        except Exception:
            pass

    # 1. Global search using Differential Evolution on smooth objective
    if time.time() < deadline - 80:
        try:
            bounds = [(-3, 3), (-0.5, 0.5), (-0.005, 0.005)]
            # Use a moderate epsilon for global search to avoid getting stuck in non-smooth local minima
            de_res = opt.differential_evolution(
                lambda c: smooth_objective(c, 1e-3), 
                bounds, 
                seed=42, 
                maxiter=150, 
                popsize=25,
                tol=1e-8, 
                mutation=(0.5, 1.5), 
                recombination=0.7,
                polish=False
            )
            if de_res.fun < best_val:
                # Verify with exact objective
                exact_val = exact_objective(de_res.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_result = de_res.x.copy()
        except Exception:
            pass

    # 2. Local refinement with decreasing epsilon
    epsilons = [1e-3, 1e-5, 1e-7, 1e-9]
    for eps in epsilons:
        if time.time() > deadline - 30:
            break
        try:
            # Use Nelder-Mead on the smooth objective
            res_smooth = opt.minimize(
                lambda c: smooth_objective(c, eps), 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_smooth.fun < best_val:
                exact_val = exact_objective(res_smooth.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_result = res_smooth.x.copy()
        except Exception:
            pass

    # 3. Final polish with exact objective
    if time.time() < deadline - 5:
        try:
            res_final = opt.minimize(
                exact_objective, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_result = res_final.x.copy()
        except Exception:
            pass
            
    # 4. Try a few diverse starting points if time remains
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        candidates = [
            x0 + rng.normal(0, 0.1, 3),
            x0 + rng.normal(0, 0.5, 3),
            np.array([-1.0, 0.05, 0.0002]),
            np.array([-1.2, 0.03, 0.0004]),
            np.array([-1.1, 0.045, 0.00028]),
        ]
        for x_start in candidates:
            if time.time() > deadline - 10:
                break
            try:
                # Quick smooth optimization
                res_r = opt.minimize(
                    lambda c: smooth_objective(c, 1e-6), 
                    x_start, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                if res_r.fun < best_val:
                    exact_val = exact_objective(res_r.x)
                    if exact_val < best_val:
                        best_val = exact_val
                        best_result = res_r.x.copy()
            except Exception:
                pass

    return best_result
# EVOLVE-BLOCK-END

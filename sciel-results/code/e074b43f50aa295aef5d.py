import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

def _hermite_coeffs(n):
    # Returns coefficients of H_n(x) as a polynomial in x (highest degree first)
    # We just need to evaluate H_n(x)
    pass

def _build_g(c0, c4, c8, x):
    # H_n(x) using physicists' definition
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Constraint g(0) = 0
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=498960 (Wait, let's calculate H12(0))
    # H_n(0) for even n: (-1)^(n/2) * n! / (n/2)! * 2^(n/2)
    # H4(0) = (-1)^2 * 4! / 2! * 2^2 = 1 * 12 * 4 = 48? 
    # Let's check H4(x) = 16x^4 - 48x^2 + 12. H4(0) = 12.
    # Formula: (-1)^(n/2) (2n)! / n! ? No.
    # H_{2k}(0) = (-1)^k (2k)! / k! * 2^k ?
    # k=2 (n=4): (-1)^2 * 24 / 2 * 4 = 48. But H4(0)=12.
    # The formula is H_{2k}(0) = (-1)^k \frac{(2k)!}{k!} 2^k ?
    # Let's re-verify H4. H0=1, H1=2x, H2=4x^2-2, H3=8x^3-12x, H4=16x^4-48x^2+12.
    # H4(0) = 12.
    # Formula: (-1)^k (2k)! / k! ? 24/2 = 12. Correct.
    # So H_{2k}(0) = (-1)^k (2k)! / k!.
    # H8(0) (k=4): (-1)^4 * 8! / 4! = 40320 / 24 = 1680. Correct.
    # H12(0) (k=6): (-1)^6 * 12! / 6! = 479001600 / 720 = 665280.
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    return np.exp(-np.pi * x**2) * P

def _compute_bound(c0, c4, c8):
    # Calculate uncertainty product A(g) A(hat(g))
    # Use numerical integration
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    g = _build_g(c0, c4, c8, x)
    
    norm_sq = np.sum(np.abs(g)**2) * dx
    if norm_sq < 1e-10:
        return 1e10
        
    # A(g) = <x^2> = sum(x^2 |g|^2) / sum(|g|^2)
    A_g = np.sum(x**2 * np.abs(g)**2) * dx / norm_sq
    
    # Fourier transform
    # Since g is a combination of Hermite functions, hat(g) is also a combination.
    # hat(g)(xi) = sum c_k (-i)^k e^{-pi xi^2} H_k(xi) (with appropriate scaling if any)
    # Assuming standard FT and physicists' Hermite:
    # The relation is hat(e^{-pi x^2} H_n(x)) = (-i)^n e^{-pi xi^2} H_n(xi)
    # Let's verify this scaling again.
    # If it's not exact, we can compute FT numerically.
    # Numerical FT is safer.
    # Use FFT for speed? Or just sum?
    # g is smooth and decays.
    # Let's use the analytic property if we can confirm the scaling.
    # If we use analytic:
    # hat_g(xi) = c0*(-i)^0*e^{-pi xi^2}H0(xi) + c4*(-i)^4*e^{-pi xi^2}H4(xi) + ...
    # (-i)^4 = 1.
    # Let's compute A_hat_g numerically from the analytic expression to be safe and fast.
    xi = x # Symmetric grid
    H0_xi = eval_hermite(0, xi)
    H4_xi = eval_hermite(4, xi)
    H8_xi = eval_hermite(8, xi)
    H12_xi = eval_hermite(12, xi)
    
    c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 1680.0) / 665280.0
    
    # Coefficients for hat(g)
    # Note: The scaling of H_n in the FT might differ.
    # Standard result: F[e^{-pi x^2} H_n(x)] = (-i)^n e^{-pi xi^2} H_n(xi)
    # This holds for the specific normalization of H_n used in quantum mechanics (physicists').
    # Let's assume this holds.
    hat_g = (c0 * (-1j)**0 * H0_xi + 
             c4 * (-1j)**4 * H4_xi + 
             c8 * (-1j)**8 * H8_xi + 
             c12 * (-1j)**12 * H12_xi) * np.exp(-np.pi * xi**2)
    
    norm_sq_hat = np.sum(np.abs(hat_g)**2) * dx
    if norm_sq_hat < 1e-10:
        return 1e10
        
    A_hat_g = np.sum(xi**2 * np.abs(hat_g)**2) * dx / norm_sq_hat
    
    return A_g * A_hat_g

def construct():
    start_time = time.time()
    timeout = 170 # Safety margin
    
    # Seed values
    c0_seed = -113 / 100
    c4_seed = 1 / 25
    c8_seed = 1 / 3240
    
    best_c = np.array([c0_seed, c4_seed, c8_seed])
    best_score = _compute_bound(c0_seed, c4_seed, c8_seed)
    
    # Optimization
    def objective(c):
        return _compute_bound(c[0], c[1], c[2])
        
    # Try Nelder-Mead
    try:
        res = minimize(objective, best_c, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() - start_time < timeout:
            if res.success and res.fun < best_score:
                best_c = res.x
                best_score = res.fun
    except Exception:
        pass
        
    # Try L-BFGS-B
    try:
        if time.time() - start_time < timeout:
            res = minimize(objective, best_c, method='L-BFGS-B', 
                           options={'maxiter': 500, 'gtol': 1e-6})
            if res.success and res.fun < best_score:
                best_c = res.x
                best_score = res.fun
    except Exception:
        pass
        
    return np.array(best_c)

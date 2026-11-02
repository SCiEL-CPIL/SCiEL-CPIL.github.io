import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit, keep safety margin
    
    # Hermite polynomials (physicists' convention, H_n)
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 24x^6 + 72x^4 - 32x^2 + 15
    # H12(x) = x^12 - 60x^10 + 300x^8 - 3200x^6 + 7200x^4 - 4320x^2 + 945
    
    # g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # g(0) = c0 + 3*c4 + 15*c8 + 945*c12 = 0
    # c12 = -(c0 + 3*c4 + 15*c8) / 945
    
    # The uncertainty bound C4 involves:
    # A(f) * A(Fourier(f)) where f = g * exp(-x^2/2) or similar
    # For the Hermite-Gaussian test function, we need to compute:
    # The bound is related to ||g||_1 / min|hat_g| where hat_g is Fourier transform
    
    # Let's define g as the polynomial part, and the full function as g(x) * exp(-x^2/2)
    # The Fourier transform of exp(-x^2/2) * H_n(x) is related to H_n in frequency domain
    
    # Actually, for the uncertainty inequality, we have:
    # A(f) = integral |f(x)| dx (L1 norm)
    # A(Fourier(f)) = integral |hat_f(xi)| dxi
    # The product A(f)*A(Fourier(f)) >= 1 (uncertainty principle)
    
    # For f(x) = g(x) * exp(-x^2/2), where g is our Hermite combination:
    # f is a finite combination of Hermite functions
    # The L1 norm can be computed numerically
    # The Fourier transform of Hermite functions are themselves (up to phases)
    
    # Let me compute numerically:
    # f(x) = g(x) * exp(-x^2/2)
    # hat_f(xi) = Fourier transform
    
    # The key is that for Hermite functions, the Fourier transform has a simple form
    # hat{psi_n}(xi) = (-i)^n psi_n(xi) for physicists' Hermite functions
    
    # So if f = sum c_n * H_n(x) * exp(-x^2/2), then
    # hat_f = sum c_n * (-i)^n * H_n(xi) * exp(-xi^2/2)
    
    # The L1 norm of f and hat_f can be computed numerically
    
    def compute_bound(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 + 3*c4 + 15*c8) / 945.0
        
        # Define g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # f(x) = g(x) * exp(-x^2/2)
        # hat_f(xi) = c0*H0(xi)*exp(-xi^2/2) + c4*(-i)^4*H4(xi)*exp(-xi^2/2) + ...
        #           = c0*H0(xi)*exp(-xi^2/2) + c4*H4(xi)*exp(-xi^2/2) + c8*(-i)^8*H8(xi)*exp(-xi^2/2) + c12*(-i)^12*H12(xi)*exp(-xi^2/2)
        # Since (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1:
        # hat_f(xi) = (c0*H0(xi) + c4*H4(xi) + c8*H8(xi) + c12*H12(xi)) * exp(-xi^2/2)
        # So hat_f has the same form as f!
        
        # This means ||f||_1 = ||hat_f||_1, so the bound is ||f||_1^2
        # Wait, that's not right for the uncertainty product
        
        # Actually, the uncertainty principle says:
        # (integral |f|^2)^(1/2) * (integral |hat_f|^2)^(1/2) >= 1/2
        # But we want L1 norms: A(f) = integral |f| dx
        
        # Let me reconsider. The task says "upper bound on A(f)*A(Fourier(f))"
        # where A is likely the L1 norm or similar
        
        # For the specific structure here, let me just compute numerically
        x = np.linspace(-15, 15, 10000)
        dx = x[1] - x[0]
        
        def H0(x): return np.ones_like(x)
        def H4(x): return x**4 - 6*x**2 + 3
        def H8(x): return x**8 - 24*x**6 + 72*x**4 - 32*x**2 + 15
        def H12(x): return x**12 - 60*x**10 + 300*x**8 - 3200*x**6 + 7200*x**4 - 4320*x**2 + 945
        
        g = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        f = g * np.exp(-x**2/2)
        
        # L1 norm of f
        norm_f = np.trapz(np.abs(f), x)
        
        # Fourier transform: since f is a combination of Hermite functions,
        # hat_f(xi) = same combination evaluated at xi (because (-i)^n = 1 for n=0,4,8,12)
        hat_f = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        hat_f = hat_f * np.exp(-x**2/2)
        
        norm_hat_f = np.trapz(np.abs(hat_f), x)
        
        return norm_f * norm_hat_f
    
    # Initial guess: the prior state of the art
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple starting points
    best_result = None
    best_score = np.inf
    
    # Starting points to try
    starts = [
        x0,
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.9, 0.06, 0.0004]),
        np.array([-1.13, 0.04, 0.0003]) + np.random.randn(3)*0.1,
        np.array([-1.13, 0.04, 0.0003]) + np.random.randn(3)*0.05,
    ]
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(compute_bound, start, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception as e:
            continue
    
    # If we found a better result, return it; otherwise return the seed
    if best_result is not None and best_score < compute_bound(x0):
        return np.array(best_result)
    else:
        return np.array([-113/100, 1/25, 1/3240])

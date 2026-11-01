import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # safety margin before 210s kill
    
    # Physicists' Hermite polynomials evaluated symbolically via coefficients
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 105840x^2 + 10395
    
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 56700*x**4 - 105840*x**2 + 10395
    
    # g(0) = a0*1 + a4*3 + a8*105 + a12*10395 = 0
    # a12 = -(a0 + 3*a4 + 105*a8) / 10395
    
    def make_g(a0, a4, a8):
        a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
        return a0, a4, a8, a12
    
    def uncertainty_product(a0, a4, a8):
        a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
        # Use Gauss-Hermite quadrature for integrals of form int f(x) e^{-x^2} dx
        # np.polynomial.hermite.hermgauss(n) gives nodes and weights for int f(x) e^{-x^2} dx
        n_quad = 100
        nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
        
        x = nodes
        w = weights
        
        H0v = H0(x)
        H4v = H4(x)
        H8v = H8(x)
        H12v = H12(x)
        
        p = a0*H0v + a4*H4v + a8*H8v + a12*H12v
        
        # ||g||^2 = int |p|^2 e^{-x^2} dx
        norm_sq = np.sum(w * p**2)
        
        # ||xg||^2 = int x^2 |p|^2 e^{-x^2} dx
        xg_norm_sq = np.sum(w * x**2 * p**2)
        
        # g'(x) = (p'(x) - x*p(x)) e^{-x^2/2}
        # p'(x) = a0*0 + a4*(4x^3 - 12x) + a8*(8x^7 - 168x^5 + 840x^3 - 840x) + a12*(12x^11 - 600x^9 + 10080x^7 - 75600x^5 + 226800x^3 - 211680x)
        dp = (a4*(4*x**3 - 12*x) + 
              a8*(8*x**7 - 168*x**5 + 840*x**3 - 840*x) + 
              a12*(12*x**11 - 600*x**9 + 10080*x**7 - 75600*x**5 + 226800*x**3 - 211680*x))
        
        gp = dp - x * p  # derivative of p(x)*e^{-x^2/2} without the exponential
        
        # ||g'||^2 = int |p' - xp|^2 e^{-x^2} dx
        gp_norm_sq = np.sum(w * gp**2)
        
        if norm_sq <= 0:
            return 1e10
        
        # Uncertainty product: sqrt(||xg||^2 * ||g'||^2) / ||g||  (normalized)
        # Or just ||xg|| * ||g'|| / ||g||^2 for the bound constant
        # Let's use the standard uncertainty product: sqrt(||xg||^2 * ||g'||^2)
        # But for the "bound on C4", it might be the ratio
        # Let me try: uncertainty product = sqrt(xg_norm_sq * gp_norm_sq) / norm_sq
        # This is ||xg|| * ||g'|| / ||g||^2 which is the standard uncertainty product for normalized functions
        
        if xg_norm_sq < 0 or gp_norm_sq < 0:
            return 1e10
        
        return np.sqrt(xg_norm_sq * gp_norm_sq) / norm_sq
    
    # Multi-start optimization
    best_val = np.inf
    best_params = None
    
    # Initial guesses
    guesses = [
        [-1.13, 0.04, 0.000309],  # seed values
        [0, 0, 0.01],
        [1, 0, 0],
        [-1, 1, -0.1],
        [0.5, -0.2, 0.05],
        [-2, 0.5, -0.05],
        [1, -0.5, 0.1],
        [-0.5, 0.2, -0.01],
        [2, -1, 0.2],
        [-3, 1, -0.15],
    ]
    
    for guess in guesses:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                lambda x: uncertainty_product(x[0], x[1], x[2]),
                np.array(guess, dtype=float),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Polish with BFGS
    if best_params is not None and time.time() < deadline - 10:
        try:
            res2 = minimize(
                lambda x: uncertainty_product(x[0], x[1], x[2]),
                best_params,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-12}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    return np.array(best_params)

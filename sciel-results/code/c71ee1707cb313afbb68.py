import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

# Hermite Polynomials (Physicists')
def H0(x):
    return np.ones_like(x) if hasattr(x, 'shape') else 1.0

def H4(x):
    if hasattr(x, 'shape'):
        return x**4 - 6*x**2 + 3
    return x**4 - 6*x**2 + 3

def H8(x):
    if hasattr(x, 'shape'):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def H12(x):
    if hasattr(x, 'shape'):
        return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 55440*x**4 - 83160*x**2 + 10395
    return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 55440*x**4 - 83160*x**2 + 10395

def g(x, c0, c4, c8):
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    return c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)

def g_prime(x, c0, c4, c8):
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    # Derivatives:
    # H0' = 0
    # H4' = 4x^3 - 12x
    # H8' = 8x^7 - 168x^5 + 840x^3 - 840x
    # H12' = 12x^11 - 600x^9 + 10080x^7 - 75600x^5 + 221760x^3 - 166320x
    dH0 = 0
    dH4 = 4*x**3 - 12*x
    dH8 = 8*x**7 - 168*x**5 + 840*x**3 - 840*x
    dH12 = 12*x**11 - 600*x**9 + 10080*x**7 - 75600*x**5 + 221760*x**3 - 166320*x
    return c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12

def objective(params):
    c0, c4, c8 = params
    
    # Integrate over a sufficiently large range
    # The Gaussian e^{-x^2/2} decays fast. 10 sigma is enough.
    limit = 10.0
    
    def integrand_I0(x):
        val = g(x, c0, c4, c8)
        return val**2 * np.exp(-x**2/2.0)
        
    def integrand_Ix(x):
        val = g(x, c0, c4, c8)
        return x**2 * val**2 * np.exp(-x**2/2.0)
        
    def integrand_Ip(x):
        gp = g_prime(x, c0, c4, c8)
        gv = g(x, c0, c4, c8)
        term = (gp - 0.5*x*gv)
        return term**2 * np.exp(-x**2/2.0)

    I0, _ = quad(integrand_I0, -limit, limit, limit=100, epsabs=1e-10, epsrel=1e-10)
    Ix, _ = quad(integrand_Ix, -limit, limit, limit=100, epsabs=1e-10, epsrel=1e-10)
    Ip, _ = quad(integrand_Ip, -limit, limit, limit=100, epsabs=1e-10, epsrel=1e-10)
    
    if I0 <= 0:
        return 1e100
        
    # Product of uncertainties
    # A(f) = sqrt(Ix/I0), A(hat f) = sqrt(Ip/I0)
    # Product = sqrt(Ix * Ip / I0^2)
    prod = np.sqrt(Ix * Ip / (I0 * I0))
    return prod

def construct():
    # Start from the seed solution
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Define a time limit guard
    start_time = time.time()
    timeout = 180.0
    safety_margin = 10.0
    
    best_params = x0
    best_score = objective(best_params)
    
    # Try a few optimization methods
    methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
    
    for method in methods:
        if time.time() - start_time > timeout - safety_margin:
            break
            
        try:
            # For L-BFGS-B, we don't provide gradients to keep it simple, but it might be slow.
            # Nelder-Mead is robust for low dimensions.
            res = minimize(objective, x0, method=method, options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success:
                new_params = res.x
                new_score = res.fun
                if new_score < best_score:
                    best_score = new_score
                    best_params = new_params
        except Exception:
            pass
            
    # If we have time, do a local refinement around the best found
    if time.time() - start_time < timeout - safety_margin:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params

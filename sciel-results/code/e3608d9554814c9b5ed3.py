import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time
import math

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
    # H12(x) = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280
    
    # Values at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Constraint: c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    def objective(c):
        c0, c4, c8 = c
        c12 = -(c0 + 12*c4 + 1680*c8) / 665280.0
        
        def g(x):
            h0 = 1.0
            h4 = 16*x**4 - 48*x**2 + 12
            h8 = 256*x**8 - 4480*x**6 + 26880*x**4 - 48384*x**2 + 1680
            h12 = (4096*x**12 - 129024*x**10 + 1548288*x**8 - 7654656*x**6 + 
                   16636928*x**4 - 15974400*x**2 + 665280)
            poly = c0*h0 + c4*h4 + c8*h8 + c12*h12
            return np.exp(-x**2/2.0) * poly
        
        # L1 norm of g (g is even)
        l1_g, _ = quad(lambda x: abs(g(x)), 0, np.inf, limit=200)
        l1_g *= 2
        
        # L1 norm of g_hat (same form due to Fourier invariance for n multiple of 4)
        l1_g_hat = l1_g
        
        # L2 norm squared
        norm_sq = np.sqrt(np.pi) * (
            c0**2 * 1 + 
            c4**2 * 384 + 
            c8**2 * 10321920 + 
            c12**2 * 1962000550400
        )
        
        if norm_sq < 1e-15:
            return np.inf
            
        return (l1_g * l1_g_hat) / norm_sq

    # Initial guess
    c_init = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_c = c_init.copy()
    best_val = objective(c_init)
    
    # Phase 1: Nelder-Mead from initial guess
    if time.time() - start_time < deadline - 30:
        try:
            res = minimize(objective, c_init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
        
    # Phase 2: Differential evolution for global search
    if time.time() - start_time < deadline - 40:
        try:
            bounds = [(-10, 10), (-2, 2), (-0.01, 0.01)]
            de_res = differential_evolution(objective, bounds, seed=42, maxiter=100, 
                                           tol=1e-10, polish=True, popsize=15,
                                           callback=lambda x, convergence: (time.time() - start_time) < deadline - 15)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_c = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine-tune with Nelder-Mead from best found
    if time.time() - start_time < deadline - 10:
        try:
            res2 = minimize(objective, best_c, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
        except Exception:
            pass
            
    return best_c

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin before 210s kill

    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
    # H12(x) = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 665280
    
    # Constraint: P(0) = 0
    # c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280

    def objective(c):
        c0, c4, c8 = c
        # c12 from constraint
        c12 = -(c0 + 12*c4 + 1680*c8) / 665280.0
        
        def g(x):
            # Vectorized evaluation for quad
            h0 = 1.0
            h4 = 16*x**4 - 48*x**2 + 12
            h8 = 256*x**8 - 4480*x**6 + 26880*x**4 - 48384*x**2 + 1680
            # H12 polynomial
            h12 = (4096*x**12 - 129024*x**10 + 1548288*x**8 - 7654656*x**6 + 
                   16636928*x**4 - 15974400*x**2 + 665280)
            poly = c0*h0 + c4*h4 + c8*h8 + c12*h12
            return np.exp(-x**2/2.0) * poly
        
        # L1 norm of g
        # g is even
        l1_g, _ = quad(lambda x: abs(g(x)), 0, np.inf, limit=200)
        l1_g *= 2
        
        # L1 norm of g_hat
        # Since n is multiple of 4, i^n = 1.
        # g_hat has same form.
        l1_g_hat = l1_g
        
        # L2 norm squared
        # ||g||_2^2 = sqrt(pi) * sum c_k^2 2^k k!
        norm_sq = np.sqrt(np.pi) * (
            c0**2 * 1 + 
            c4**2 * 384 +  # 2^4 * 4! = 16 * 24 = 384
            c8**2 * 10321920 +  # 2^8 * 8! = 256 * 40320
            c12**2 * 1962000550400  # 2^12 * 12! = 4096 * 479001600
        )
        
        if norm_sq < 1e-15:
            return np.inf
            
        return (l1_g * l1_g_hat) / norm_sq

    # Initial guess
    c_init = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_c = c_init
    best_val = objective(c_init)
    
    # Optimization phase 1: Nelder-Mead from initial point
    try:
        res = minimize(objective, c_init, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    except Exception:
        pass
        
    # Check time
    if time.time() - start_time < deadline - 10:
        # Try a few random restarts with Nelder-Mead
        rng = np.random.default_rng(42)
        for _ in range(5):
            c_rand = c_init + rng.normal(0, 0.1, 3)
            try:
                res_r = minimize(objective, c_rand, method='Nelder-Mead', 
                                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_c = res_r.x
            except Exception:
                pass
            if time.time() - start_time > deadline - 5:
                break
                
    # Optimization phase 2: Differential Evolution for global search
    if time.time() - start_time < deadline - 30:
        # Define bounds for the coefficients
        # c0 is likely around -1 to -2
        # c4 is likely around 0 to 0.1
        # c8 is likely very small
        bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
        
        try:
            # Use differential_evolution to explore the space more broadly
            # maxiter is set low to ensure we don't timeout, but enough for some exploration
            # tol is tight for precision
            de_res = differential_evolution(objective, bounds, maxiter=50, 
                                             popsize=15, tol=1e-8, 
                                             seed=42, polish=True, 
                                             init='latinhypercube')
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_c = de_res.x
        except Exception:
            pass
            
    return best_c

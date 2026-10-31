import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time
import math

def construct():
    start_time = time.time()
    deadline = 180.0 # Safety margin
    
    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 4480x^6 + 26880x^4 - 48384x^2 + 1680
    # H12(x) = 4096x^12 - 129024x^10 + 1548288x^8 - 7654656x^6 + 16636928x^4 - 15974400x^2 + 1995840
    
    # Coefficients at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 1995840.0
    
    # Note: H4(0) = 12, H8(0) = 1680, H12(0) = 1995840.
    # Wait, H_n(0) = (-1)^{n/2} (n-1)!! 
    # H4(0) = (-1)^2 3!! = 3*1 = 3? 
    # Let's check H4(x) = 16x^4 - 48x^2 + 12. H4(0)=12.
    # Formula: H_{2k}(0) = (-1)^k (2k)!/k!.
    # k=2: (-1)^2 4!/2! = 24/2 = 12. Correct.
    # k=4: (-1)^4 8!/4! = 40320/24 = 1680. Correct.
    # k=6: (-1)^6 12!/6! = 479001600/720 = 665280? 
    # 12!/6! = 665280.
    # My previous manual calc for H12(0) was -1995840.
    # Let's re-verify H12(0).
    # 12! = 479001600. 6! = 720.
    # 479001600 / 720 = 665280.
    # So H12(0) = 665280.
    # Let's check the polynomial expansion of H12(0).
    # H12(x) = ... + 1995840?
    # 1995840 / 665280 = 3.
    # There is a discrepancy.
    # Let's use the formula $H_{2k}(0) = (-1)^k \frac{(2k)!}{k!}$.
    # H12(0) = 665280.
    
    # Constraint: c0*1 + c4*12 + c8*1680 + c12*665280 = 0?
    # Wait, H4(0)=12. H8(0)=1680.
    # c0 + 12c4 + 1680c8 + 665280c12 = 0.
    # c12 = -(c0 + 12c4 + 1680c8) / 665280.
    
    # Let's re-verify the seed coefficients with this constraint.
    # c0 = -1.13, c4 = 0.04, c8 = 1/3240 ~ 0.0003086.
    # c0 + 12c4 + 1680c8 = -1.13 + 0.48 + 0.518 = -0.132.
    # c12 = 0.132 / 665280 ~ 2e-7.
    # This seems consistent.
    
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
    
    # Optimization phase 1: Nelder-Mead from initial guess
    try:
        res = minimize(objective, c_init, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    except Exception:
        pass
        
    # Optimization phase 2: Differential evolution for global search
    if time.time() - start_time < deadline - 30:
        try:
            bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
            de_res = differential_evolution(objective, bounds, seed=42, maxiter=50, 
                                           tol=1e-10, polish=True, 
                                           callback=lambda x, convergence: (time.time() - start_time) < deadline - 10)
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_c = de_res.x
        except Exception:
            pass
    
    # Optimization phase 3: Fine-tune with Nelder-Mead from best found
    if time.time() - start_time < deadline - 10:
        try:
            res2 = minimize(objective, best_c, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x
        except Exception:
            pass
            
    return best_c

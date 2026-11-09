"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import time

# Pre-compute Hermite polynomials
# Physicists' Hermite polynomials
H0 = sp.hermite(0)
H4 = sp.hermite(4)
H8 = sp.hermite(8)
H12 = sp.hermite(12)

# Values at 0 for constraint P(0)=0
# H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
H0_0 = H0(0)
H4_0 = H4(0)
H8_0 = H8(0)
H12_0 = H12(0)

def construct():
    start_time = time.time()
    deadline = 150.0 # Safety margin
    
    # Grid for integration
    # The function decays as e^{-x^2/4}. x=10 is e^{-25} ~ 1e-11.
    # 10000 points is sufficient.
    x = np.linspace(-10, 10, 10000)
    exp_term = np.exp(-x**2 / 4.0)
    
    def calculate_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def objective(coeffs):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        
        # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        P = c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)
        
        # Integral of |P(x)| e^{-x^2/4}
        integrand = np.abs(P) * exp_term
        integral = np.trapz(integrand, x)
        
        return integral

    # Initial guess from seed
    # Seed: [-113/100, 1/25, 1/3240]
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = None
    best_val = np.inf
    
    # Try L-BFGS-B
    try:
        res = opt.minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'maxfun': 1000})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    except Exception:
        pass
    
    # Try Powell if time remains
    if time.time() < deadline - 10:
        try:
            res2 = opt.minimize(objective, x0, method='Powell', 
                                options={'maxiter': 500, 'maxfev': 1000})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x
        except Exception:
            pass
            
    # Try a few random restarts if time remains
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            x_rand = rng.normal(0, 1, 3)
            # Scale to be reasonable
            x_rand = x_rand * np.array([1.0, 0.1, 0.01])
            try:
                res_r = opt.minimize(objective, x_rand, method='Nelder-Mead', 
                                     options={'maxiter': 200})
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_result = res_r.x
            except Exception:
                pass

    if best_result is None:
        return x0
    return best_result
# EVOLVE-BLOCK-END

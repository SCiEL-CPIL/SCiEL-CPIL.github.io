import numpy as np
from scipy.special import hermite
from scipy.optimize import minimize
from scipy.integrate import quad
import time

# Precompute Hermite polynomials
# Note: scipy.special.hermite returns physicists' Hermite polynomials
H0 = hermite(0)
H4 = hermite(4)
H8 = hermite(8)
H12 = hermite(12)

# Values at 0
H0_0 = H0(0)
H4_0 = H4(0)
H8_0 = H8(0)
H12_0 = H12(0)

def get_c12(c0, c4, c8):
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

def objective(coeffs):
    c0, c4, c8 = coeffs
    c12 = get_c12(c0, c4, c8)
    
    # Define f(x)
    def f(x):
        return (c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)) * np.exp(-0.5 * x**2)
    
    # Calculate L2 norm squared
    # Integrate |f(x)|^2
    integrand_l2 = lambda x: f(x)**2
    l2_sq, _ = quad(integrand_l2, -np.inf, np.inf, limit=100)
    
    if l2_sq <= 1e-10:
        return 1e10 # Avoid division by zero
        
    # Calculate L4 norm
    # Integrate |f(x)|^4
    integrand_l4 = lambda x: f(x)**4
    l4, _ = quad(integrand_l4, -np.inf, np.inf, limit=100)
    
    # Objective: L4 / L2^2
    return l4 / (l2_sq * l2_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 190.0 # Safety margin
    
    best_score = float('inf')
    best_coeffs = np.array([-1.13, 0.04, 0.00030864]) # Initial guess from seed
    
    # Initial seed from the current program
    current_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    current_score = objective(current_seed)
    if current_score < best_score:
        best_score = current_score
        best_coeffs = current_seed

    # Perform local optimizations from multiple starting points
    # We will try a grid of random starting points and run Nelder-Mead
    rng = np.random.default_rng(42)
    
    # Define a range for coefficients. The seed has c0 ~ -1, c4 ~ 0.04, c8 ~ 0.0003.
    # Let's sample in a reasonable range.
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate a random starting point
        # Scale based on expected magnitudes
        c0_init = rng.uniform(-2.0, 2.0)
        c4_init = rng.uniform(-0.2, 0.2)
        c8_init = rng.uniform(-0.01, 0.01)
        
        x0 = np.array([c0_init, c4_init, c8_init])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    # If we have time, do a final polish with L-BFGS-B from the best point
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_coeffs, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    # Return the coefficients c0, c4, c8
    return best_coeffs

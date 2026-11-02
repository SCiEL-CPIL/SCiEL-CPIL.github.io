import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Precompute grid
# Use a sufficiently large range and fine step for accurate L1 norm
x = np.linspace(-10, 10, 10001)
dx = x[1] - x[0]
gauss = np.exp(-x**2 / 2)

# Probabilists' Hermite polynomials He_0, He_4, He_8, He_12 evaluated on x
def he0(x): return np.ones_like(x)
def he4(x): return x**4 - 6*x**2 + 3
def he8(x): return x**8 - 28*x**6 + 420*x**4 - 840*x**2 + 105
def he12(x): return x**12 - 66*x**10 + 2310*x**8 - 39900*x**6 + 332640*x**4 - 798336*x**2 + 10395

H0 = he0(x)
H4 = he4(x)
H8 = he8(x)
H12 = he12(x)

# Values at 0
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

def c3_from(c0, c1, c2):
    return -(c0 + H4_0*c1 + H8_0*c2) / H12_0

def objective(coeffs):
    """
    Computes the square of the L1 norm of the function g(x).
    The score is the upper bound, which is proportional to (L1 norm)^2.
    """
    c0, c1, c2 = coeffs
    c3 = c3_from(c0, c1, c2)
    
    # Compute g(x)
    poly = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = poly * gauss
    
    # L1 norm
    l1 = np.sum(np.abs(g)) * dx
    
    return l1**2

def construct():
    t0 = time.time()
    deadline = t0 + 170  # 10s safety margin
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective(best)
    
    # 1. Global Search with Differential Evolution
    # Bounds: The coefficients shouldn't be extremely large. 
    # Based on seed, values are around -1, 0.04, 0.0003.
    # Let's allow a wide range to ensure we don't miss the global min.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    try:
        if time.time() < deadline - 60:
            # DE is stochastic, so set a seed for reproducibility if needed, but we want exploration
            res_de = differential_evolution(objective, bounds, maxiter=100, popsize=20, seed=42, tol=1e-8)
            if res_de.fun < best_val:
                best = res_de.x
                best_val = res_de.fun
    except Exception:
        pass
    
    # 2. Local Refinement
    # Use Powell's method which is derivative-free and robust
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(objective, best, method='Powell', options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12})
            if res_powell.fun < best_val:
                best = res_powell.x
                best_val = res_powell.fun
        except Exception:
            pass
        
    # 3. Final Polish with Nelder-Mead (often good for final convergence)
    if time.time() < deadline - 2:
        try:
            res_nm = minimize(objective, best, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res_nm.fun < best_val:
                best = res_nm.x
                best_val = res_nm.fun
        except Exception:
            pass
            
    return np.array(best)

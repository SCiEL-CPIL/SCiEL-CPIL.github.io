import numpy as np
import time
from scipy.optimize import minimize

# Precompute grid
x = np.linspace(-8, 8, 4001)
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
    c0, c1, c2 = coeffs
    c3 = c3_from(c0, c1, c2)
    g = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * gauss
    l1 = np.sum(np.abs(g)) * dx
    return l1**2  # or l1

def construct():
    t0 = time.time()
    deadline = t0 + 180  # safety margin
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    best = seed.copy()
    best_val = objective(best)
    
    # Try optimization
    try:
        # Nelder-Mead from seed
        res = minimize(objective, seed, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol':1e-10, 'fatol':1e-12})
        if time.time() < deadline and res.fun < best_val:
            best = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Maybe multi-start
    if time.time() < deadline:
        rng = np.random.default_rng(0)
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            start = seed + rng.normal(0, 0.5, size=3)
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                               options={'maxiter': 2000})
                if time.time() > deadline - 30:
                    break
                if res.fun < best_val:
                    best = res.x
                    best_val = res.fun
            except Exception:
                pass
    
    return np.array(best)

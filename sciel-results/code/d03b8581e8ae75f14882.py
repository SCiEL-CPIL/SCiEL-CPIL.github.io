import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    H_nm2 = np.ones_like(x)
    H_nm1 = 2 * x
    for k in range(2, n + 1):
        H_n = 2 * x * H_nm1 - 2 * (k - 1) * H_nm2
        H_nm2 = H_nm1
        H_nm1 = H_n
    return H_n

def _g_and_coeffs(a0, a4, a8):
    """Return coefficients for H0, H4, H8, H12 with g(0)=0 constraint."""
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    a12 = -(a0 * 1 + a4 * 12 + a8 * 1680) / 665280.0
    return a0, a4, a8, a12

def _objective(coeffs):
    """Compute the upper bound (proxy: L1 norm of g squared, since FT = g for these degrees)."""
    a0, a4, a8, a12 = _g_and_coeffs(coeffs[0], coeffs[1], coeffs[2])
    
    x = np.linspace(-20, 20, 40001)
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gauss
    
    A_f = np.trapz(np.abs(g), x)
    return A_f ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 175  # safety margin
    
    seed_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    seed_val = _objective(seed_coeffs)
    
    best_val = seed_val
    best_coeffs = seed_coeffs.copy()
    
    rng = np.random.default_rng(42)
    
    # Multi-start local optimization
    starts = []
    for scale in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for _ in range(5):
            pert = rng.normal(0, scale, size=3)
            starts.append(seed_coeffs + pert)
    for _ in range(50):
        starts.append(rng.uniform(-20, 20, size=3))
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Differential evolution
    if time.time() < deadline - 40:
        try:
            bounds = [(-50, 50), (-50, 50), (-50, 50)]
            de_res = differential_evolution(_objective, bounds, maxiter=80, 
                                           popsize=12, seed=42, tol=1e-10,
                                           polish=True, init='sobol')
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            final_res = minimize(_objective, best_coeffs, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            if final_res.fun < best_val:
                best_val = final_res.fun
                best_coeffs = final_res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)

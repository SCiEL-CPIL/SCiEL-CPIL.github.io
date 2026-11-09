"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    # Physics of the Hermite-Gaussian test function
    # H0(0)=1, H4(0)=12, H8(0)=1344, H12(0)=145152 (physicist's convention)
    # Norm factor: ||H_k * e^{-x^2/2}||^2 = 2^k * k!
    # For C4 upper bound: minimize A(g)*A(Fourier(g))
    
    # Key: for g = sum a_k H_k e^{-x^2/2}, the L2 norm squared is sum (a_k^2 * 2^k * k!)
    # The Fourier transform of H_k e^{-x^2/2} is i^k H_k e^{-x^2/2} (up to conventions)
    # So A(g) = A(ĝ) = sqrt(sum a_k^2 2^k k!)
    # C4 upper bound ~ A(g)^2 = sum a_k^2 2^k k!
    
    # But wait - the constraint g(0)=0 means:
    # a0*1 + a4*12 + a8*1344 + a12*145152 = 0
    # So a12 = -(a0 + 12*a4 + 1344*a8) / 145152
    
    # The objective to minimize is the upper bound on C4, which for this structure
    # is proportional to the L2 norm squared: sum_k a_k^2 * 2^k * k!
    
    def compute_objective(a):
        a0, a4, a8 = a
        a12 = -(a0 + 12.0*a4 + 1344.0*a8) / 145152.0
        # L2 norm squared = sum a_k^2 * 2^k * k!
        obj = (a0**2 * 1 + 
               a4**2 * 2**4 * 24 +   # 16 * 24 = 384
               a8**2 * 2**8 * 40320 + # 256 * 40320 = 10321920
               a12**2 * 2**12 * 479001600)  # 4096 * 479001600
        return obj
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_obj = compute_objective(seed)
    
    # Multi-start Nelder-Mead optimization
    deadline = time.time() + 180
    best_a = seed.copy()
    best_obj = seed_obj
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    start_points = [seed.copy()]
    
    # Add random perturbations around seed
    for i in range(20):
        if time.time() > deadline - 30:
            break
        perturb = rng.randn(3) * 0.1
        sp = seed + perturb
        start_points.append(sp)
    
    # Add some structured starts
    for scale in [0.5, 1.0, 2.0]:
        for signs in [(1,1,1), (1,-1,1), (-1,1,-1), (1,1,-1)]:
            if time.time() > deadline - 30:
                break
            sp = np.array([signs[0]*scale, signs[1]*scale*0.01, signs[2]*scale*0.001])
            start_points.append(sp)
    
    for sp in start_points:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(compute_objective, sp, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_a = res.x.copy()
        except Exception:
            pass
    
    # If we have time, do a finer local search
    if time.time() < deadline - 10:
        try:
            res = minimize(compute_objective, best_a, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_a = res.x.copy()
        except Exception:
            pass
    
    return best_a
# EVOLVE-BLOCK-END

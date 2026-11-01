import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 180  # 180s limit, keep margin
    
    # Hermite polynomials values at 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    # Weights w_i = ||phi_i||_2^2 proportional to 2^i i!
    # We can use log weights to avoid overflow, or just regular floats if we scale
    # w_i = sqrt(pi) * 2^i * i!
    # Let's compute log_w
    import math
    log_w = np.zeros(13)
    for i in range(13):
        log_w[i] = math.log(2) * i + math.lgamma(i + 1)
        
    # E_i = i + 0.5
    E = np.array([i + 0.5 for i in range(13)])
    
    indices = [0, 4, 8, 12]
    w_idx = np.exp(log_w[indices])
    E_idx = E[indices]
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        c = np.array([c0, c4, c8, c12])
        
        # Numerator: sum c_i^2 E_i w_i
        num = np.sum(c**2 * E_idx * w_idx)
        # Denominator: sum c_i^2 w_i
        den = np.sum(c**2 * w_idx)
        
        if den < 1e-10:
            return 1e10
        return num / den

    # Initial guesses
    # 1. Seed
    x0_seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    # 2. Analytical guess (c8=0, c12=0 => c0=-3c4)
    x0_guess = np.array([-3.0, 1.0, 0.0])
    
    best_val = float('inf')
    best_x = x0_seed
    
    # Evaluate initial points
    for x0 in [x0_seed, x0_guess]:
        val = objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0
            
    # Run Nelder-Mead
    # Use multiple starts if time permits
    starts = [x0_seed, x0_guess]
    
    # Add some random perturbations around the best guess
    np.random.seed(42)
    for _ in range(5):
        pert = np.random.normal(0, 0.5, size=3)
        starts.append(x0_guess + pert)
        
    for x0 in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final polish with higher precision if time allows
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return np.array(best_x)

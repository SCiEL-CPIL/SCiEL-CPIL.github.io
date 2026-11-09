import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_herm
import time

def construct():
    start_time = time.time()
    deadline = start_time + 170  # safety margin for 180s limit
    
    # Precompute on a fine grid
    x = np.linspace(-40, 40, 50001)
    dx = x[1] - x[0]
    exp_neg_x2 = np.exp(-x**2/2)
    H0 = np.ones_like(x)
    H4 = eval_herm(4, x)
    H8 = eval_herm(8, x)
    H12 = eval_herm(12, x)
    
    def get_g(a0, a4, a8):
        a12 = -(a0 + 12*a4 + 1680*a8) / 1032192.0
        P = a0*H0 + a4*H4 + a8*H8 + a12*H12
        g = P * exp_neg_x2
        return g
    
    def objective(params):
        a0, a4, a8 = params
        g = get_g(a0, a4, a8)
        l2_sq = np.sum(g**2) * dx
        if l2_sq <= 1e-30:
            return 1e10
        g_norm = g / np.sqrt(l2_sq)
        l1 = np.sum(np.abs(g_norm)) * dx
        return l1
    
    # Seed value
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_val = objective(seed)
    
    best_val = seed_val
    best_params = seed.copy()
    
    # Multiple starting points for robustness
    np.random.seed(42)
    starts = [seed.copy()]
    # Perturbations around seed
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(5):
            s = seed + np.random.randn(3) * scale * np.array([0.2, 0.02, 0.0002])
            starts.append(s)
    
    for i, s in enumerate(starts):
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead', 
                          options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)

import numpy as np
import time
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients for H0, H4, H8, H12
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 122880x^10 + 1397760x^8 - 6812160x^6 + 13977600x^4 - 11827200x^2 + 4084080

def compute_value(coeffs):
    c0, c1, c2 = coeffs
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    A = -48.0 * c1 - 13440.0 * c2 - 11827200.0 * c3
    B = 16.0 * c1 + 11520.0 * c2 + 13977600.0 * c3
    C = -3072.0 * c2 - 6812160.0 * c3
    D = 256.0 * c2 + 1397760.0 * c3
    E = -122880.0 * c3
    F = 4096.0 * c3
    
    coeffs_q = [F, E, D, C, B, A]
    
    # Find roots of q(y)
    roots = np.roots(coeffs_q)
    
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10:
            r = root.real
            if r > 1e-12:
                eps = max(1e-8, abs(r) * 1e-8)
                q_minus = np.polyval(coeffs_q, r - eps)
                q_plus = np.polyval(coeffs_q, r + eps)
                if q_minus * q_plus < 0:
                    valid_roots.append(r)
            
    if not valid_roots:
        return 1e9
        
    r_max = max(valid_roots)
    return r_max ** 2 / (2.0 * np.pi)

def _g_values(coeffs, x_grid):
    """Evaluate g(x) at points in x_grid. Returns array."""
    c0, c1, c2 = coeffs
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    x2 = x_grid ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    x8 = x4 ** 2
    x10 = x8 * x2
    x12 = x8 * x4
    
    g = (c0 * 1.0 +
         c1 * (16.0 * x4 - 48.0 * x2 + 12.0) +
         c2 * (256.0 * x8 - 3072.0 * x6 + 11520.0 * x4 - 13440.0 * x2 + 1680.0) +
         c3 * (4096.0 * x12 - 122880.0 * x10 + 1397760.0 * x8 - 6812160.0 * x6 + 13977600.0 * x4 - 11827200.0 * x2 + 4084080.0))
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = compute_value(seed)
    best_val = seed_val
    best_coeffs = seed.copy()
    
    # Use a log-spaced grid in y = x^2 for better resolution near the root
    # y ranges from 0 to some max. The root r is in y-space, x = sqrt(y)
    # We'll use y_grid from 1e-6 to 100
    n_y = 2000
    y_grid = np.geomspace(1e-6, 100.0, n_y)
    x_grid = np.sqrt(y_grid)
    
    # Also add points very close to 0
    y_grid_fine = np.concatenate([np.linspace(0, 1e-4, 50), y_grid])
    x_grid_fine = np.sqrt(y_grid_fine)
    
    beta = 50.0  # Sharpness of softplus penalty
    
    def objective(params):
        """Minimize r^2 with smooth positivity penalty via log-sum-exp."""
        c0, c1, c2 = params
        coeffs = [c0, c1, c2]
        
        # Compute the actual value (this is the target to minimize)
        val = compute_value(coeffs)
        
        # Smooth positivity penalty using log-sum-exp
        # We want to penalize g(x) < 0
        # Use t = -beta * g(x), penalty = log-sum-exp(t) / beta
        # This is smooth and approximates max(0, -g(x)) when beta is large
        
        g_vals = _g_values(coeffs, x_grid_fine)
        t = -beta * g_vals
        # Stable log-sum-exp
        t_max = np.max(t)
        if t_max > 50:
            penalty = (t_max + np.log(np.sum(np.exp(t - t_max)))) / beta
        else:
            penalty = np.log(np.sum(np.exp(t))) / beta
        
        return val + penalty
    
    # Multi-start Nelder-Mead optimization
    rng = np.random.default_rng(42)
    n_starts = 80
    
    # Create a set of initial points
    start_points = [seed.copy()]
    
    # Add points near the seed with decreasing step sizes
    for scale in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]:
        for i in range(10):
            if time.time() > deadline - 30.0:
                break
            start_points.append(seed + rng.normal(0, scale, size=3))
    
    # Add some random points
    for i in range(30):
        if time.time() > deadline - 30.0:
            break
        start_points.append(rng.uniform(-2.0, 2.0, size=3))
    
    for start in start_points:
        if time.time() > deadline - 15.0:
            break
        
        try:
            # Use Nelder-Mead with tighter tolerances
            res = minimize(objective, start, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
            
            if res.fun < best_val:
                # Verify the result is valid (g >= 0) with a tight check
                g_check = _g_values(res.x, x_grid_fine)
                min_g = np.min(g_check)
                if min_g >= -1e-6:  # allow tiny numerical error
                    # Polish: run a very tight optimization from this point
                    res_polish = minimize(objective, res.x, method='Nelder-Mead',
                                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12,
                                                   'adaptive': True})
                    if res_polish.fun < best_val:
                        g_polish = _g_values(res_polish.x, x_grid_fine)
                        if np.min(g_polish) >= -1e-6:
                            best_val = res_polish.fun
                            best_coeffs = res_polish.x.copy()
        except Exception:
            pass
    
    # Final check: if best is still seed, return seed
    if best_val >= seed_val:
        return seed
    
    return best_coeffs

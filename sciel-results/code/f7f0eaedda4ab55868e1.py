"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s hard limit

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution: (h*h)[k] = sum_i h[i]*h[k-i]
    # For non-negative h, the supremum of the autoconvolution is at k=0
    # if h is concentrated, but generally we need to compute it.
    # However, for non-negative h, max autoconvolution is at the peak of the sum.
    # Let's compute the full autoconvolution to be safe.
    h_conv = np.convolve(h, h, mode='full')
    l_inf = np.max(h_conv)
    if l_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _negative_ratio(h):
    """Objective function for minimization (negative of ratio)."""
    return -_ratio(h)

# EVOLVE-BLOCK-START
def construct():
    # Structural change: Use numerical optimization to find better configurations.
    # The ratio is scale-invariant, so we can fix L1 norm = 1 without loss of generality.
    # We optimize over the shape of h.
    
    # Start with a geometric decay profile, which is known to perform well.
    # h[i] = r^i, normalized to L1=1.
    # Try a few different decay rates and use them as starting points.
    
    best_h = None
    best_score = 0.0
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return [0.0] * N
    
    # Generate initial candidates
    candidates = []
    
    # 1. Delta function
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    candidates.append(h_delta)
    
    # 2. Geometric decays with different rates
    for r in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        h_geom = np.array([r**i for i in range(N)])
        h_geom /= np.sum(h_geom)
        candidates.append(h_geom)
    
    # 3. Gaussian-like profiles
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        x = np.arange(N) - N//2
        h_gauss = np.exp(-x**2 / (2*sigma**2))
        h_gauss /= np.sum(h_gauss)
        candidates.append(h_gauss)
    
    # 4. Uniform
    h_unif = np.ones(N) / N
    candidates.append(h_unif)
    
    # Evaluate all candidates
    for h in candidates:
        score = _ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return list(best_h) if best_h is not None else [0.0] * N
    
    # Use Nelder-Mead to refine the best candidate
    # We optimize over log-spaced parameters or direct values
    # Since h must be non-negative and L1-normalized, we can use a softmax parameterization
    # or simply optimize directly with bounds.
    
    # Use direct optimization with bounds [0, 1]
    # Fix L1 norm by normalizing after each step
    
    def _objective_normalized(params):
        h = np.asarray(params, dtype=np.float64)
        h = np.maximum(h, 0.0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        h = h / l1
        return -_ratio(h)
    
    # Start from the best candidate found
    if best_h is not None:
        x0 = best_h.copy()
        bounds = [(0, 1)] * N
        
        # Try Nelder-Mead optimization
        try:
            result = minimize(_objective_normalized, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if time.time() - _START_TIME < _TIME_LIMIT:
                h_opt = np.maximum(result.x, 0.0)
                l1 = np.sum(h_opt)
                if l1 > 1e-15:
                    h_opt = h_opt / l1
                    score_opt = _ratio(h_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return list(best_h) if best_h is not None else [0.0] * N
    
    # Try a few more random restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Random sparse vector
        h_rand = np.zeros(N)
        support_size = np.random.randint(3, 15)
        indices = np.random.choice(N, support_size, replace=False)
        h_rand[indices] = np.random.exponential(1.0, support_size)
        h_rand /= np.sum(h_rand)
        
        try:
            result = minimize(_objective_normalized, h_rand, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if time.time() - _START_TIME < _TIME_LIMIT:
                h_opt = np.maximum(result.x, 0.0)
                l1 = np.sum(h_opt)
                if l1 > 1e-15:
                    h_opt = h_opt / l1
                    score_opt = _ratio(h_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    if best_h is None:
        return [0.0] * N
    
    return list(best_h)
# EVOLVE-BLOCK-END

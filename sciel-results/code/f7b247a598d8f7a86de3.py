"""Second autocorrelation inequality: maximize L2^2 / (L1 * max_autoconv)"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _compute_ratio(h):
    """Compute R = sum(h^2) / (sum(h) * max(autoconv))"""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return -1.0
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    auto = np.convolve(h, h, mode='full')
    max_auto = np.max(auto)
    if max_auto < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_auto)

def _neg_ratio(h):
    """Negative ratio for minimization"""
    r = _compute_ratio(h)
    if r < 0:
        return 1e10
    return -r

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

# EVOLVE-BLOCK-START
def construct():
    """
    Maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) over nonnegative h in R^50.
    
    Key observations:
    - R is NOT scale invariant: scaling h by alpha scales R by 1/alpha
    - So we want small-magnitude h with favorable shape
    - But h must be nonnegative, and the ratio depends on the shape
    - The shape that minimizes max_auto relative to L2^2/L1 is what matters
    
    Strategy: Use multi-start local optimization with various initial patterns.
    Since R increases as magnitude decreases, we'll normalize to a reasonable scale
    and focus on finding the optimal shape, then scale appropriately.
    
    Actually, since R = L2^2/(L1*max_auto) and this is homogeneous of degree -1,
    the optimal shape is what matters, and we can pick any scale. Let's fix L1=1
    and maximize L2^2/max_auto, then the scale doesn't matter for the shape.
    """
    best_h = None
    best_r = 0.0
    
    # Fix scale: set L1 = 1 to remove scale degree of freedom
    # Then maximize L2^2 / max_auto subject to sum(h)=1, h>=0
    
    def _ratio_fixed_l1(h):
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, 0.0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return -1e10
        h_norm = h / l1
        l2_sq = np.sum(h_norm * h_norm)
        auto = np.convolve(h_norm, h_norm, mode='full')
        max_auto = np.max(auto)
        if max_auto < 1e-15:
            return -1e10
        return l2_sq / max_auto
    
    def _neg_ratio_fixed_l1(h):
        r = _ratio_fixed_l1(h)
        if r < 0:
            return 1e10
        return -r
    
    # Try various initial patterns
    initial_patterns = []
    
    # Uniform
    initial_patterns.append(np.ones(N) / N)
    
    # Single spike
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        initial_patterns.append(h)
    
    # Two adjacent spikes
    for i in range(N - 1):
        h = np.zeros(N)
        h[i] = 0.5
        h[i+1] = 0.5
        initial_patterns.append(h)
    
    # Two separated spikes
    for gap in [2, 5, 10, 20, 25]:
        for i in range(N - gap):
            h = np.zeros(N)
            h[i] = 0.5
            h[i+gap] = 0.5
            initial_patterns.append(h)
    
    # Triangular
    for center in [0, N//4, N//2, 3*N//4, N-1]:
        h = np.zeros(N)
        for j in range(N):
            h[j] = max(0, 1 - abs(j - center) / (N/2))
        if np.sum(h) > 0:
            h /= np.sum(h)
            initial_patterns.append(h)
    
    # Gaussian-like
    for center in [N//4, N//2, 3*N//4]:
        for sigma in [2, 5, 10, 15, 25]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            if np.sum(h) > 0:
                h /= np.sum(h)
                initial_patterns.append(h)
    
    # Cosine bump
    for center in [N//4, N//2, 3*N//4]:
        for width in [5, 10, 20, 30, 40]:
            h = np.zeros(N)
            x = np.arange(N)
            mask = np.abs(x - center) < width/2
            h[mask] = 0.5 * (1 + np.cos(2 * np.pi * (x[mask] - center) / width))
            if np.sum(h) > 0:
                h /= np.sum(h)
                initial_patterns.append(h)
    
    # Random patterns
    rng = np.random.RandomState(42)
    for _ in range(20):
        h = rng.random(N)
        if np.sum(h) > 0:
            h /= np.sum(h)
            initial_patterns.append(h)
    
    # Sparse random
    for _ in range(20):
        k = rng.randint(2, 10)
        indices = rng.choice(N, k, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0 / k
        initial_patterns.append(h)
    
    # Optimize each initial pattern
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
    ]
    bounds = [(0.0, None) for _ in range(N)]
    
    for i, h0 in enumerate(initial_patterns):
        if _time_remaining() < 2.0:
            break
        
        # Evaluate initial
        r0 = _ratio_fixed_l1(h0)
        if r0 > best_r:
            best_r = r0
            best_h = h0.copy()
        
        # Local optimization
        try:
            result = minimize(
                _neg_ratio_fixed_l1,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if result.success or result.fun < 0:
                h_opt = np.maximum(result.x, 0.0)
                if np.sum(h_opt) > 0:
                    h_opt /= np.sum(h_opt)
                    r_opt = _ratio_fixed_l1(h_opt)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Additional refinement: try Nelder-Mead on top candidates
    if best_h is not None and _time_remaining() > 5.0:
        # Create perturbed versions of best
        rng2 = np.random.RandomState(123)
        candidates = [best_h.copy()]
        for _ in range(10):
            h_pert = best_h.copy()
            h_pert += rng2.normal(0, 0.1, N)
            h_pert = np.maximum(h_pert, 0.0)
            if np.sum(h_pert) > 0:
                h_pert /= np.sum(h_pert)
                candidates.append(h_pert)
        
        for h0 in candidates:
            if _time_remaining() < 1.0:
                break
            try:
                result = minimize(
                    _neg_ratio_fixed_l1,
                    h0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                h_opt = np.maximum(result.x, 0.0)
                if np.sum(h_opt) > 0:
                    h_opt /= np.sum(h_opt)
                    r_opt = _ratio_fixed_l1(h_opt)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final: convert back to original scale
    # Since R = L2^2/(L1*max_auto) and we fixed L1=1, we have R_shape = L2^2/max_auto
    # The actual R for h with L1=1 is exactly R_shape
    # So we just return the optimized h (with L1=1)
    
    if best_h is None:
        # Fallback: single spike
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    # Ensure nonnegative and return as list
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END

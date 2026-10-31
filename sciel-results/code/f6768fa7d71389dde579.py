import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Normalize to L1=1
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # The ratio is ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Under L1=1 normalization, this is ||h||_2^2 / ||h*h||_inf
    # 
    # Key insight: We want to maximize the L2 norm squared (concentration)
    # while minimizing the peak of the autoconvolution.
    # 
    # For a delta function h=[1,0,...,0]: ||h||_2^2=1, ||h*h||_inf=1, ratio=1
    # For uniform h=[1/N,...,1/N]: ||h||_2^2=1/N, ||h*h||_inf=1/N, ratio=1
    # 
    # The maximum possible value is bounded by 1 (by Cauchy-Schwarz / Young's inequality).
    # But the inspirations suggest scores > 1 are achievable, which means the evaluator
    # might be using a different normalization or the discrete approximation matters.
    #
    # Let's use scipy's minimize to directly optimize this non-convex objective.
    # We'll use multiple random restarts with different initializations.
    
    def neg_ratio(params):
        """Negative ratio for minimization."""
        h = np.maximum(params, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 1.0  # Large penalty
        h_norm = h / l1
        h2_sq = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        conv_inf = np.max(conv)
        if conv_inf < 1e-15:
            return 1.0
        return -h2_sq / conv_inf
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    # Generate diverse initial points
    np.random.seed(42)
    initial_points = []
    
    # 1. Uniform
    initial_points.append(np.ones(N) / N)
    
    # 2. Delta functions
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        initial_points.append(h.copy())
    
    # 3. Sparse random
    for _ in range(50):
        k = np.random.randint(1, 10)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        weights = np.random.dirichlet(np.ones(k) * np.random.uniform(0.1, 5.0))
        h[idx] = weights
        initial_points.append(h.copy())
    
    # 4. Structured patterns
    for period in [2, 3, 5, 7]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = 1.0 if (i % period == 0) else 0.0
        if np.sum(h) > 0:
            h /= np.sum(h)
            initial_points.append(h.copy())
    
    # 5. Gaussian bumps
    for center in range(0, N, 3):
        for width in [1, 2, 3, 5]:
            x = np.arange(N)
            h = np.exp(-((x - center)**2) / (2 * width**2))
            if np.sum(h) > 0:
                h /= np.sum(h)
                initial_points.append(h.copy())
    
    # 6. Random Dirichlet
    for _ in range(50):
        alpha = np.ones(N) * np.random.uniform(0.1, 10.0)
        h = np.random.dirichlet(alpha)
        initial_points.append(h.copy())
    
    # Sort by initial score and try top candidates with optimization
    scored_points = [(compute_ratio(h), h) for h in initial_points]
    scored_points.sort(reverse=True)
    
    # Use L-BFGS-B with bounds [0, 1] for non-negativity
    bounds = [(0.0, 1.0)] * N
    
    for score, h0 in scored_points[:30]:
        if time.time() > deadline - 10:
            break
        
        # Try optimization from this starting point
        try:
            result = minimize(neg_ratio, h0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.success or result.fun < neg_ratio(h0):
                h_opt = np.maximum(result.x, 0)
                l1 = np.sum(h_opt)
                if l1 > 1e-15:
                    h_opt /= l1
                    opt_score = compute_ratio(h_opt)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_h = h_opt.copy()
        except Exception:
            continue
    
    # If we haven't improved much, try a more aggressive approach:
    # Optimize over a smaller support set
    if best_score < 1.05 and time.time() < deadline - 20:
        for k in [1, 2, 3, 5, 8]:
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                idx = np.random.choice(N, k, replace=False)
                h0 = np.zeros(N)
                weights = np.random.dirichlet(np.ones(k) * np.random.uniform(0.5, 2.0))
                h0[idx] = weights
                
                # Optimize only over the support
                def neg_ratio_subset(params):
                    h = np.zeros(N)
                    h[idx] = np.maximum(params, 0)
                    l1 = np.sum(h)
                    if l1 < 1e-15:
                        return 1.0
                    h_norm = h / l1
                    h2_sq = np.sum(h_norm**2)
                    conv = np.convolve(h_norm, h_norm, mode='full')
                    conv_inf = np.max(conv)
                    if conv_inf < 1e-15:
                        return 1.0
                    return -h2_sq / conv_inf
                
                try:
                    result = minimize(neg_ratio_subset, weights, method='L-BFGS-B',
                                     bounds=[(0.0, 1.0)] * k,
                                     options={'maxiter': 500, 'ftol': 1e-12})
                    if result.success:
                        h_opt = np.zeros(N)
                        h_opt[idx] = np.maximum(result.x, 0)
                        l1 = np.sum(h_opt)
                        if l1 > 1e-15:
                            h_opt /= l1
                            opt_score = compute_ratio(h_opt)
                            if opt_score > best_score:
                                best_score = opt_score
                                best_h = h_opt.copy()
                except Exception:
                    continue
    
    return list(best_h)

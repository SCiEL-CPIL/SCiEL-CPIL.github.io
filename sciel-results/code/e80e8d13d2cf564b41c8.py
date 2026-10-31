"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Strategy: Deterministic multi-start L-BFGS-B on a smooth surrogate.
    # The objective is the squared ratio of max to min distance.
    # Direct optimization of max/min is non-smooth. We use a smooth approximation:
    # f(x) = (sum exp(alpha * d_sq))^(1/alpha) approximates max d_sq as alpha -> inf
    # g(x) = (sum exp(-alpha * d_sq))^(1/alpha) approximates min d_sq as alpha -> inf
    # Ratio^2 = (max d / min d)^2 = (max d_sq) / (min d_sq)
    # We minimize log(max_d_sq) - log(min_d_sq) approximated by:
    # LSE(alpha, d_sq) - LSE(-alpha, d_sq)
    
    # To make it stable, we use a moderate alpha and optimize the log-ratio.
    # Let d_sq be the squared distances.
    # Objective: log( sum exp(alpha * d_sq) / alpha ) - log( sum exp(-alpha * d_sq) / alpha )
    # This is smooth and differentiable.
    
    alpha = 50.0  # Sharpness of max/min approximation
    
    def objective_smooth(x):
        pts = x.reshape(N, 2)
        # Normalize to keep points in a reasonable range, though L-BFGS-B on normalized 
        # coordinates might be tricky if normalization is non-differentiable at 0.
        # Instead, we optimize raw coordinates but penalize large scales or just rely on 
        # the ratio being scale-invariant. The gradient of the ratio w.r.t scale is 0.
        # However, numerical stability is better if points are centered.
        # Let's center the points inside the objective for stability, but the gradient 
        # calculation must account for this or we just optimize centered coordinates.
        # Simplest: Optimize raw coords. The ratio is scale invariant.
        
        d_sq = get_sq_dists(pts)
        
        # Avoid numerical overflow in exp
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # If any distance is 0, return large penalty
        if min_d_sq < 1e-12:
            return 1e10
        
        # Log-sum-exp for max approximation
        # log(sum exp(alpha * d_sq)) = alpha * max_d_sq + log(sum exp(alpha * (d_sq - max_d_sq)))
        log_sum_exp_max = alpha * max_d_sq + np.log(np.sum(np.exp(alpha * (d_sq - max_d_sq))))
        
        # Log-sum-exp for min approximation
        # log(sum exp(-alpha * d_sq)) = -alpha * min_d_sq + log(sum exp(-alpha * (d_sq - min_d_sq)))
        log_sum_exp_min = -alpha * min_d_sq + np.log(np.sum(np.exp(-alpha * (d_sq - min_d_sq))))
        
        # Objective: log(max_approx) - log(min_approx)
        # max_approx = exp(log_sum_exp_max) / alpha ? No, LSE is log(sum exp).
        # We want to approximate (max_d / min_d)^2 = max_d_sq / min_d_sq.
        # log(max_d_sq / min_d_sq) = log(max_d_sq) - log(min_d_sq).
        # Using LSE: log(sum exp(alpha d_sq)) ~ alpha max_d_sq + log(N_pairs).
        # So log(max_d_sq) ~ (log_sum_exp_max - log(N_pairs)) / alpha.
        # Similarly for min.
        
        n_pairs = N * (N - 1) // 2
        log_max_approx = (log_sum_exp_max - np.log(n_pairs)) / alpha
        log_min_approx = (log_sum_exp_min - np.log(n_pairs)) / (-alpha) # Wait, -alpha is in exp
        
        # Correction:
        # sum exp(-alpha * d_sq) ~ exp(-alpha * min_d_sq) * N_pairs
        # log(sum exp(-alpha * d_sq)) = -alpha * min_d_sq + log(N_pairs)
        # log_min_approx = - (log_sum_exp_min - np.log(n_pairs)) / alpha
        
        val = log_max_approx - log_min_approx
        return val

    def gradient_smooth(x):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        
        if np.min(d_sq) < 1e-12:
            return np.zeros_like(x)
        
        n_pairs = N * (N - 1) // 2
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # We need the gradient of the objective w.r.t x.
        # This is complex to derive analytically for LSE.
        # Given the time constraint and complexity, using finite differences or 
        # relying on L-BFGS-B's internal numerical gradient might be safer/faster 
        # to implement correctly. However, analytical gradients are much faster.
        
        # Let's try a simpler smooth objective:
        # Minimize (mean(d_sq^2) / mean(1/d_sq^2))? No.
        
        # Alternative: Use the "soft" max/min directly with a moderate alpha 
        # and let L-BFGS-B handle numerical gradients. With 32 dimensions, 
        # numerical gradients are feasible.
        
        return None # Force numerical gradient

    # Create diverse initial configurations
    initial_configs = []
    
    # 1. Random points
    for _ in range(5):
        pts = np.random.uniform(-2, 2, (N, 2))
        initial_configs.append(normalize(pts).ravel())
    
    # 2. Perturbed 4x4 grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    base_grid = normalize(base_grid)
    for _ in range(5):
        noise = np.random.normal(0, 0.1, N*2)
        perturbed = base_grid.ravel() + noise
        initial_configs.append(normalize(perturbed.reshape(N, 2)).ravel())
    
    # 3. Circle-based configurations
    for radius in [1.0, 1.5, 2.0]:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        noise = np.random.normal(0, 0.1, (N, 2))
        pts = pts + noise
        initial_configs.append(normalize(pts).ravel())
        
    # 4. Two clusters
    for _ in range(3):
        pts1 = np.random.uniform(-1, 1, (8, 2))
        pts2 = np.random.uniform(-1, 1, (8, 2))
        pts = np.vstack([pts1 - 2, pts2 + 2])
        initial_configs.append(normalize(pts).ravel())

    best_overall_x = None
    best_overall_score = 1e10

    # Run L-BFGS-B from each initial configuration
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        # We optimize the raw score directly? No, raw score is non-smooth.
        # We optimize the smooth surrogate.
        # But L-BFGS-B needs a function. Let's use the smooth objective.
        # To ensure we get a good raw score, we evaluate raw score at the end.
        
        try:
            res = minimize(
                objective_smooth,
                config,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            # Evaluate raw score of the result
            raw_score = score_raw(res.x.reshape(N, 2))
            
            if raw_score < best_overall_score:
                best_overall_score = raw_score
                best_overall_x = res.x.copy()
                
        except Exception as e:
            continue

    # Final polish with Nelder-Mead on the raw score to handle the non-smoothness
    if best_overall_x is not None:
        def objective_raw(x):
            return score_raw(x.reshape(N, 2))
        
        try:
            res = minimize(
                objective_raw,
                best_overall_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < best_overall_score:
                best_overall_score = res.fun
                best_overall_x = res.x
        except Exception:
            pass

    final_pts = best_overall_x.reshape(N, 2)
    return final_pts

# EVOLVE-BLOCK-END

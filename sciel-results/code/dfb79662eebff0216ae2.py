"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])
    
    def log_score_scaled(pts):
        """
        Compute log of the score Delta/n^n where Delta is product of scaled distances.
        Score = (prod_{i!=j} d_ij_scaled) / n^n
        d_ij_scaled = d_ij * (2 / D)
        Log Score = sum_{i!=j} (log(d_ij) + log(2) - log(D)) - n log(n)
                   = 2 * sum_{i<j} log(d_ij) + n(n-1)(log(2) - log(D)) - n log(n)
        """
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        # Avoid log(0)
        d_ij_safe = np.maximum(d_ij, 1e-15)
        D = np.max(dists)
        D_safe = max(D, 1e-15)
        
        log_prod_unordered = np.sum(np.log(d_ij_safe))
        # 2 * sum_{i<j} log(d_ij) is sum_{i!=j} log(d_ij)
        log_delta_raw = 2.0 * log_prod_unordered
        # Scaling factor contribution
        log_scale_factor = n * (n - 1) * (np.log(2.0) - np.log(D_safe))
        
        log_score_val = log_delta_raw + log_scale_factor - n * np.log(n)
        return log_score_val, D
    
    def neg_log_score_scaled(pts):
        """Negative log score for minimization."""
        ls, _ = log_score_scaled(pts)
        return -ls
    
    def neg_log_score_grad(pts):
        """
        Gradient of -log_score_scaled.
        L = 2 * sum_{i<j} log(d_ij) - n(n-1) log(D) + C
        dL/dp_k = 2 * sum_{j != k} (p_k - p_j) / d_kj^2 - n(n-1) * (1/D) * dD/dp_k
        
        dD/dp_k = max_{j != k} ( (p_k - p_j) / d_kj )_component-wise? No.
        D = max_{i<j} d_ij.
        The gradient of max is the gradient of the maximizing pair(s).
        If unique max pair (u, v), dD/dp_u = (p_u - p_v)/d_uv, dD/dp_v = (p_v - p_u)/d_uv.
        """
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        D = np.max(dists)
        
        # Gradient of 2 * sum_{i<j} log(d_ij)
        grad_log_prod = np.zeros_like(pts)
        for idx in range(num_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = d_ij[idx]
            if d < 1e-15:
                continue
            diff = pts[i] - pts[j]
            # d/dp_i (2 log d) = 2 * diff / d^2
            # d/dp_j (2 log d) = -2 * diff / d^2
            grad_log_prod[i] += 2.0 * diff / (d * d)
            grad_log_prod[j] -= 2.0 * diff / (d * d)
            
        # Gradient of -n(n-1) log(D)
        # We need dD/dp_k
        grad_D = np.zeros_like(pts)
        max_val = -1
        # Find all pairs that achieve the max diameter
        # Note: dists is symmetric
        max_pairs = []
        for idx in range(num_pairs):
            i, j = iu[0][idx], iu[1][idx]
            if np.isclose(dists[i,j], D, rtol=1e-9, atol=1e-12):
                max_pairs.append((i, j))
                
        if max_pairs:
            # Average the gradients of the max pairs if there are multiple
            for i, j in max_pairs:
                d = dists[i,j]
                if d < 1e-15:
                    continue
                diff = pts[i] - pts[j]
                # d/dp_i d_ij = diff/d
                # d/dp_j d_ij = -diff/d
                grad_D[i] += diff / d
                grad_D[j] -= diff / d
            # Normalize if multiple pairs?
            # Technically subgradient is convex hull. Averaging is a valid subgradient.
            if len(max_pairs) > 1:
                grad_D /= len(max_pairs)
                
        grad_penalty = -n * (n - 1) / D * grad_D
        
        total_grad = grad_log_prod + grad_penalty
        return total_grad
        
    def objective_and_grad(x):
        if time.time() > deadline - 5:
            return np.inf, np.ones_like(x)
        pts = x.reshape(n, 2)
        val, _ = log_score_scaled(pts)
        return -val, -neg_log_score_grad(pts).flatten()

    # Bounds for DE: points should be within a reasonable range
    # Diameter is 2, so max coordinate spread is 2.
    # Centering isn't enforced, so bounds can be [-2, 2] or similar.
    # Let's use [-2, 2] for each coordinate.
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    best_score = -np.inf
    best_points = None
    
    # Stage 1: Differential Evolution
    # Use a subset of the time for global search
    de_deadline = start_time + 120.0
    
    try:
        # seed for reproducibility in local tests, but random in production might be better?
        # Using a fixed seed ensures deterministic behavior for evaluation if needed, 
        # but random is better for exploration. Let's use random.
        res_de = differential_evolution(
            lambda x: neg_log_score_scaled(x.reshape(n, 2)),
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.9,
            seed=None,
            polish=False, # We will polish manually
            init='sobol'
        )
        
        if res_de.success or res_de.fun < np.inf:
            opt_pts = res_de.x.reshape(n, 2)
            ls, _ = log_score_scaled(opt_pts)
            if ls > best_score:
                best_score = ls
                best_points = opt_pts.copy()
                
    except Exception as e:
        pass # If DE fails, fall back to other methods
    
    # Stage 2: Local Polish from DE result and other initializations
    if best_points is not None:
        x0 = best_points.flatten()
    else:
        # Fallback to regular octagon
        theta = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])
        x0 = best_points.flatten()
        
    # Polish with L-BFGS-B
    if time.time() < deadline - 10:
        try:
            res_polish = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if res_polish.fun < np.inf:
                opt_pts = res_polish.x.reshape(n, 2)
                ls, _ = log_score_scaled(opt_pts)
                if ls > best_score:
                    best_score = ls
                    best_points = opt_pts.copy()
        except Exception:
            pass
            
    # Stage 3: Try a few random restarts if time permits
    if time.time() < deadline - 30:
        rng = np.random.RandomState(42)
        for i in range(5):
            if time.time() > deadline - 15:
                break
            # Random initialization in box
            x0 = rng.uniform(-2, 2, (n, 2))
            try:
                res_rand = minimize(
                    lambda x: objective_and_grad(x)[0],
                    x0.flatten(),
                    jac=lambda x: objective_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
                )
                if res_rand.fun < np.inf:
                    opt_pts = res_rand.x.reshape(n, 2)
                    ls, _ = log_score_scaled(opt_pts)
                    if ls > best_score:
                        best_score = ls
                        best_points = opt_pts.copy()
            except Exception:
                continue

    # Ensure best_points is valid
    if best_points is None:
        theta = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Final check for distinct points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                best_points[j] += np.array([1e-8, 1e-8])
                
    # Ensure coordinates are within bounds (they should be, but just in case)
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points * (1e5 / max_coord)
        
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END

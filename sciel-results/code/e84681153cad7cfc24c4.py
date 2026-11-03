"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(points):
    """Compute the packing score: min of half-min-pair-dist and min boundary margin."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Boundary margin: min over i,k of (0.5 - |c_ik|)
    margin = 0.5 - np.max(np.abs(points), axis=1)
    min_margin = np.min(margin)
    if min_margin < -1e-12:
        # Invalid: some point outside
        # But we still compute a score for optimization purposes (penalty)
        pass
    # Pairwise distances
    # Use chunked computation to save memory
    min_pair = np.inf
    for i in range(N):
        diffs = points[i + 1:] - points[i]
        d = np.linalg.norm(diffs, axis=1)
        if d.size > 0:
            m = np.min(d)
            if m < min_pair:
                min_pair = m
    half_pair = min_pair / 2.0
    return min(half_pair, min_margin)


def _grad_score(points):
    """
    Compute gradient of the score w.r.t. points.
    Score = min(half_pair_dist, boundary_margin)
    We use a subgradient approach: gradient of the minimum is the average of gradients
    of active constraints (those within a small tolerance of the minimum).
    """
    pts = points.copy()
    margin = 0.5 - np.max(np.abs(pts), axis=1)
    min_margin = np.min(margin)
    
    # Pairwise min
    min_pair = np.inf
    min_pair_idx = (-1, -1)
    for i in range(N):
        diffs = pts[i + 1:] - pts[i]
        d = np.linalg.norm(diffs, axis=1)
        if d.size > 0:
            idx = np.argmin(d)
            if d[idx] < min_pair:
                min_pair = d[idx]
                min_pair_idx = (i, i + 1 + idx)
    half_pair = min_pair / 2.0
    
    score = min(half_pair, min_margin)
    
    grad = np.zeros_like(pts)
    tol = 1e-6 * max(1.0, abs(score))
    
    # Boundary gradient contribution
    if abs(min_margin - score) < tol:
        # For each point, the active boundary constraint is the one with min margin
        # d/dx_k of (0.5 - |x_k|) = -sign(x_k) for the active coordinate
        for i in range(N):
            if abs(margin[i] - min_margin) < tol:
                # Which coordinate is active?
                ax = np.abs(pts[i])
                if ax[0] >= ax[1]:
                    grad[i, 0] += -np.sign(pts[i, 0])
                else:
                    grad[i, 1] += -np.sign(pts[i, 1])
    
    # Pairwise gradient contribution
    if abs(half_pair - score) < tol:
        i, j = min_pair_idx
        if i >= 0:
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d > 1e-12:
                unit = diff / d
                # d/dx_i of (|x_i - x_j| / 2) = unit / 2
                # d/dx_j of (|x_i - x_j| / 2) = -unit / 2
                grad[i] += 0.5 * unit
                grad[j] += -0.5 * unit
    
    return grad


def _project(points):
    """Project points into the square [-0.5, 0.5]^2."""
    pts = points.copy()
    np.clip(pts, -0.5, 0.5, out=pts)
    return pts


def construct():
    t_start = time.time()
    deadline = t_start + 120.0  # Safety margin before 150s kill
    
    # Initial guess: hexagonal-ish grid
    # For 200 points, use a grid with slight perturbation
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].astype(float)
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    pts += np.random.randn(N, 2) * 0.001
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # L-BFGS-like optimization using scipy
    from scipy.optimize import minimize
    
    def objective(x):
        p = x.reshape(N, 2)
        # We want to maximize score, so minimize negative score
        # But score is nonsmooth; use a smooth approximation
        # Soft-min: -logsumexp(-a * values) / a approximates min
        # Compute all pairwise distances and boundary margins
        # This is expensive, so use a chunked approach
        
        # Boundary margins
        margins = 0.5 - np.max(np.abs(p), axis=1)
        
        # Pairwise distances (chunked)
        # For smooth surrogate, we need all pairwise distances
        # Use a soft-min over all pairwise half-distances and margins
        
        # To keep it fast, sample a subset of pairs if needed
        # For now, compute all pairs in chunks
        
        chunk_size = 50
        log_sum_exp_vals = []
        a = 200.0  # Sharpness parameter
        
        # Add margins
        for m_val in margins:
            log_sum_exp_vals.append(-a * m_val)
        
        # Add pairwise half-distances
        for i in range(N):
            diffs = p[i + 1:] - p[i]
            d = np.linalg.norm(diffs, axis=1) / 2.0
            for d_val in d:
                log_sum_exp_vals.append(-a * d_val)
        
        # Soft min = -logsumexp(-a * vals) / a
        # Compute logsumexp stably
        vals = np.array(log_sum_exp_vals)
        max_val = np.max(vals)
        if not np.isfinite(max_val):
            return 0.0
        logsumexp = max_val + np.log(np.sum(np.exp(vals - max_val)))
        soft_min = -logsumexp / a
        
        # We want to maximize soft_min, so minimize -soft_min
        return -soft_min
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Run optimization with time check
    result = None
    try:
        # First, a quick run to see if it helps
        res = minimize(objective, pts.flatten(), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
        if time.time() > deadline - 30:
            # Not enough time, return best so far
            p_res = res.x.reshape(N, 2)
            s_res = _score(p_res)
            if s_res > best_score:
                best_pts = p_res
                best_score = s_res
            return best_pts
        
        if res.x is not None:
            p_res = res.x.reshape(N, 2)
            s_res = _score(p_res)
            if s_res > best_score:
                best_pts = p_res
                best_score = s_res
        
        # Second, more iterations if time permits
        if time.time() < deadline - 30:
            res2 = minimize(objective, best_pts.flatten(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res2.x is not None:
                p_res2 = res2.x.reshape(N, 2)
                s_res2 = _score(p_res2)
                if s_res2 > best_score:
                    best_pts = p_res2
                    best_score = s_res2
        
        # Third, even more if time permits
        if time.time() < deadline - 20:
            res3 = minimize(objective, best_pts.flatten(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res3.x is not None:
                p_res3 = res3.x.reshape(N, 2)
                s_res3 = _score(p_res3)
                if s_res3 > best_score:
                    best_pts = p_res3
                    best_score = s_res3
                    
    except Exception as e:
        pass
    
    # Final projection to ensure validity
    best_pts = _project(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END

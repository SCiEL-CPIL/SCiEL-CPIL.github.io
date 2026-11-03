"""Optimized packing of 300 equal circles in a unit circle using scipy least_squares."""
import numpy as np
import time
from scipy.optimize import least_squares

N = 300
TIME_LIMIT = 115.0  # evaluator kills at 150s; keep a safety margin

def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    # Best known r ~ 0.05315 -> spacing ~ 0.1063. Use a slightly finer grid.
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(ys)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        # fallback: enlarge grid
        return _hex_init_fallback(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()

def _hex_init_fallback(n=N):
    spacing = 0.08
    k = int(1.2 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()

def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    # pairwise distances
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)

def _residuals(pts, target_r):
    """
    Construct residuals for least_squares.
    We want to maximize min(d_ij, b_i).
    We minimize sum of squared residuals.
    
    For pairs (i, j): residual = target_r - d_ij.
    If d_ij > target_r, residual is negative. Squaring makes it positive.
    This penalizes distances that are less than target_r heavily.
    However, standard least squares penalizes large deviations.
    
    A better formulation for "maximize the minimum" is to ensure all constraints are satisfied.
    If we set target_r slightly above the current best, we push distances up.
    
    Let's use:
    r_pair = target_r - d_ij
    r_bound = target_r - b_i
    
    We want these to be <= 0 for a feasible solution with radius target_r.
    Minimizing sum(r^2) will try to make them 0.
    If we increase target_r, the optimizer tries to increase distances to match.
    
    To make it smooth and robust:
    We minimize the sum of squares of the "violations" plus a small constant to avoid zero gradients?
    No, if d_ij > target_r, residual is negative. Square is positive.
    If we want to maximize the minimum, we can just minimize the sum of squares of (target_r - d_ij) for all pairs and (target_r - b_i) for all boundaries.
    
    Wait, if d_ij is very large, residual is very negative, square is huge.
    This means the optimizer will try to make d_ij close to target_r, not just >= target_r.
    This is actually good for packing: it forces uniform spacing (crystalline structure) rather than allowing large gaps.
    It effectively minimizes the variance of distances around target_r while pushing the minimum up.
    
    However, we must be careful. If we start with a hex lattice, distances are roughly uniform.
    We can start with a target_r slightly larger than the current min distance.
    """
    n = len(pts)
    # Pairwise distances
    # Vectorized calculation of all pairwise distances is O(N^2) memory and time.
    # For N=300, 300x300 matrix is 90,000 entries. This is fine.
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    
    # Residuals for pairs: target_r - d
    # We only care about pairs where d < target_r + margin?
    # No, least squares handles it. But large d gives large residual.
    # To prevent the optimizer from collapsing distant pairs to target_r (which might hurt local packing),
    # we can weight the residuals.
    # Actually, if d >> target_r, residual is large negative. Square is large positive.
    # The gradient will push d down towards target_r.
    # This is BAD. We want d >= target_r. We don't want to pull distant points closer.
    
    # Correction: We only want to penalize if d < target_r.
    # But "max(0, target_r - d)" is not differentiable at 0.
    # We can use a smooth approximation: softplus(target_r - d) or similar?
    # Or simply: if we use least_squares, we can define residuals as:
    # r = target_r - d
    # But only for pairs where d < target_r * 1.5?
    # Or better: Use a "hinge" loss approximation.
    
    # Alternative: Minimize sum of (max(0, target_r - d))^2.
    # This is differentiable almost everywhere.
    # Let's use the standard approach for such problems:
    # Residuals are (target_r - d) for all pairs, but we can scale them.
    # Actually, the "pulling together" of distant points is a feature if we want a tight packing?
    # No, in a circle packing, you want the points to fill the space.
    # If two points are far apart, they are not the bottleneck.
    # The bottleneck is the minimum distance.
    
    # Let's try a different objective:
    # Minimize sum of (target_r - d_ij)^2 for all pairs where d_ij < target_r + delta.
    # This is complex to vectorize efficiently with dynamic mask.
    
    # Simpler robust method:
    # Residuals = target_r - d_ij for ALL pairs.
    # This minimizes the variance of distances around target_r.
    # Combined with boundary constraints, it tends to form a lattice.
    # If we increase target_r, it pushes the minimum up.
    # The risk is "collapsing" the outer shell.
    # But boundary residuals (target_r - b_i) will push points outward if b_i < target_r.
    
    # Let's stick to:
    # r_pair = target_r - d_ij
    # r_bound = target_r - b_i
    # This minimizes the mean squared error of distances from target_r.
    # It effectively maximizes the minimum distance by forcing all distances to be at least target_r (if possible) and as close as possible.
    
    r_pair = target_r - d
    np.fill_diagonal(r_pair, 0.0)
    
    # Boundary
    rad = np.hypot(pts[:, 0], pts[:, 1])
    b = 1.0 - rad
    r_bound = target_r - b
    
    # Concatenate
    return np.concatenate([r_pair.flatten(), r_bound])

def _optimize(pts, start_time, time_limit):
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Start with a target radius slightly larger than current best
    current_r = best_score
    target_r = current_r * 1.01
    
    # We will iteratively increase target_r
    # Each iteration, we run least_squares to satisfy the new target.
    
    max_iter = 50
    iter_count = 0
    
    while iter_count < max_iter:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Define residuals function for scipy
        def residuals(x):
            p = x.reshape(n, 2)
            return _residuals(p, target_r)
        
        # Initial guess
        x0 = pts.flatten()
        
        # Run least_squares
        # 'trf' is robust. 'lm' is faster but requires full rank Jacobian.
        # With N=300, 600 variables, and ~45,000 residuals, Jacobian is large.
        # 'trf' uses finite differences or analytic Jacobian.
        # We don't provide analytic Jacobian, so it uses finite differences.
        # This might be slow.
        # Let's check if we can provide Jacobian? No, too complex for this snippet.
        # Finite differences on 600 vars with 45k residuals: 600 * 45k * 2 ops ~ 54M ops per eval.
        # Might be too slow for many iterations.
        
        # Optimization: Only compute residuals for "active" constraints?
        # Least squares doesn't support that directly.
        
        # Alternative: Use a subset of residuals?
        # No, we need all pairs to prevent overlap.
        
        # Let's try running it with a tight max_nfev to see if it's feasible.
        try:
            res = least_squares(residuals, x0, method='trf', max_nfev=100, xtol=1e-12, ftol=1e-12)
            new_pts = res.x.reshape(n, 2)
            
            # Ensure validity
            rad = np.hypot(new_pts[:, 0], new_pts[:, 1])
            out = rad > 1.0
            if np.any(out):
                new_pts[out] = new_pts[out] / rad[out, None]
                
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts
                # Increase target_r slightly
                target_r = new_score * 1.02
                iter_count += 1
            else:
                # No improvement, break
                break
                
        except Exception as e:
            # If optimization fails, break
            break
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    pts = _hex_init(N)
    # ensure valid start: clip inside
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    base_score = _score(pts)
    best_pts = pts.copy()
    best_score = base_score

    try:
        opt_pts, opt_score = _optimize(pts, start_time, TIME_LIMIT)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    except Exception:
        pass

    # final safety: ensure finite, inside, shape (N,2)
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        # rebuild
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]

    return best_pts

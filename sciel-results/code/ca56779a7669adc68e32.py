"""Improved circle packing: SDP-inspired initialization + binary search on radius with projected gradient descent."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time

N = 50
SEED = 42


def _pairwise_dists(pts):
    """Compute pairwise distances for N x 2 array."""
    return squareform(pdist(pts))


def _compute_radius(pts):
    """Compute the feasible radius for a set of centers."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    if pts.shape != (N, 2):
        return 0.0
    # Check all centers are within [-0.5, 0.5]^2
    if np.any(pts < -0.5 - 1e-12) or np.any(pts > 0.5 + 1e-12):
        return 0.0
    # Distance to boundary
    min_boundary = np.min(0.5 - np.abs(pts))
    # Min pairwise distance / 2
    d = _pairwise_dists(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return float(min(min_boundary, min_pair))


def _soft_min(distances, t):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    # distances: 1D array of positive values
    # Return softmin = -log(sum(exp(-d_i / t))) * t
    d = distances / t
    d_min = np.min(d)
    return -t * (np.log(np.sum(np.exp(-(d - d_min)))) - d_min)


def _soft_min_grad(distances, t):
    """Gradient of soft_min w.r.t. the distances array."""
    d = distances / t
    d_min = np.min(d)
    exp_terms = np.exp(-(d - d_min))
    weights = exp_terms / np.sum(exp_terms)
    return -weights


def _projected_gradient_step(pts, r, step_size, t):
    """One projected gradient descent step for feasibility at radius r."""
    d = _pairwise_dists(pts)
    np.fill_diagonal(d, 0.0)
    
    # Gradient of soft min of (d_ij - 2r)^+ for pairs with d_ij < 2r
    mask = d < 2 * r
    if not np.any(mask):
        # All pair constraints satisfied, check boundary
        min_boundary = np.min(0.5 - np.abs(pts))
        if min_boundary >= r:
            return pts.copy()
        # Gradient from boundary violation
        grad = np.zeros_like(pts)
        for i in range(N):
            for k in range(2):
                if pts[i, k] > 0.5 - r:
                    grad[i, k] += 1.0
                elif pts[i, k] < -0.5 + r:
                    grad[i, k] -= 1.0
        return pts - step_size * grad
    
    # For pairs where d < 2r, compute soft min gradient
    # We use a simple approach: for each pair violating, push apart
    grad = np.zeros_like(pts)
    
    # Weighted sum of violation gradients
    # For pair (i,j) with d < 2r, the gradient of (d - 2r)^2 / 2 w.r.t. x_i is (x_i - x_j) / d
    # We use soft weights to avoid singularities
    for i in range(N):
        for j in range(i + 1, N):
            if d[i, j] < 2 * r and d[i, j] > 1e-12:
                diff = pts[i] - pts[j]
                w = (2 * r - d[i, j]) / d[i, j]
                grad[i] += w * diff
                grad[j] -= w * diff
    
    # Boundary constraints
    for i in range(N):
        for k in range(2):
            if pts[i, k] > 0.5 - r:
                grad[i, k] += 1.0
            elif pts[i, k] < -0.5 + r:
                grad[i, k] -= 1.0
    
    new_pts = pts - step_size * grad
    # Project back into box [-0.5+r, 0.5-r] if r > 0
    if r > 0:
        new_pts = np.clip(new_pts, -0.5 + r, 0.5 - r)
    else:
        new_pts = np.clip(new_pts, -0.5, 0.5)
    return new_pts


def _feaspd(pts, r, max_iter=2000, step_size=None):
    """Projected gradient descent to find a feasible configuration at radius r."""
    if step_size is None:
        step_size = 0.1 / N
    cur = pts.copy()
    for it in range(max_iter):
        d = _pairwise_dists(cur)
        np.fill_diagonal(d, 0.0)
        min_pair = np.min(d) / 2.0
        min_boundary = np.min(0.5 - np.abs(cur))
        cur_r = min(min_pair, min_boundary)
        if cur_r >= r - 1e-12:
            return cur
        # Adaptive step size
        cur = _projected_gradient_step(cur, r, step_size, 1.0)
    return cur


def _construct_initial():
    """Create an initial configuration using a hexagonal-like packing."""
    # Use a hexagonal lattice for initial guess
    # For 50 circles, try to fit as many as possible in hexagonal pattern
    # Hexagonal packing: rows offset by half spacing
    # Start with a grid and perturb
    
    # Try multiple initial configurations and pick the best
    best_pts = None
    best_r = 0.0
    
    rng = np.random.RandomState(SEED)
    
    for trial in range(10):
        # Hexagonal lattice initialization
        m = int(np.ceil(np.sqrt(N)))
        # Spacing
        s = 1.0 / m
        pts = np.zeros((N, 2))
        idx = 0
        for row in range(m + 2):
            y = -0.5 + s / 2 + row * s
            offset = 0.0 if row % 2 == 0 else s / 2
            for col in range(m + 2):
                if idx >= N:
                    break
                x = -0.5 + s / 2 + col * s + offset
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts[idx] = [x, y]
                    idx += 1
            if idx >= N:
                break
        if idx < N:
            # Fill remaining with random points
            for i in range(idx, N):
                pts[i] = rng.uniform(-0.4, 0.4, size=2)
        
        # Small random perturbation
        pts += rng.uniform(-0.01, 0.01, size=pts.shape)
        pts = np.clip(pts, -0.49, 0.49)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts, best_r


def _binary_search_radius(pts_init, time_budget):
    """Binary search on radius with feasibility checks via projected gradient descent."""
    start_time = time.time()
    
    # Lower bound: current radius
    lo = _compute_radius(pts_init)
    # Upper bound: theoretical max ~ 1/sqrt(50) * pi/6 related, but use a safe upper bound
    hi = 0.08  # Best known is ~0.0714
    
    best_pts = pts_init.copy()
    best_r = lo
    
    # Binary search iterations
    for it in range(30):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        # Try to find feasible config at radius mid
        cur = _feaspd(pts_init, mid, max_iter=500, step_size=0.05 / N)
        cur_r = _compute_radius(cur)
        
        if cur_r >= mid - 1e-12:
            lo = mid
            if cur_r > best_r:
                best_r = cur_r
                best_pts = cur.copy()
        else:
            hi = mid
    
    return best_pts, best_r


def construct():
    """Construct centers for 50 equal circles in unit square [-0.5, 0.5]^2."""
    total_start = time.time()
    deadline = 130.0  # Safety margin from 150s kill
    
    # Step 1: Get a good initial configuration
    pts_init, r_init = _construct_initial()
    
    # Step 2: Binary search on radius
    time_left = deadline - (time.time() - total_start)
    if time_left < 5:
        return pts_init
    
    best_pts, best_r = _binary_search_radius(pts_init, time_left * 0.6)
    
    # Step 3: Local refinement with finer steps
    time_left = deadline - (time.time() - total_start)
    if time_left > 10:
        # Try multiple restarts from perturbed best
        rng = np.random.RandomState(SEED + 1)
        for restart in range(5):
            if time.time() - total_start > deadline - 5:
                break
            pts_pert = best_pts + rng.uniform(-0.005, 0.005, size=best_pts.shape)
            pts_pert = np.clip(pts_pert, -0.49, 0.49)
            
            # Local optimization
            step_size = 0.02 / N
            for _ in range(1000):
                if time.time() - total_start > deadline - 5:
                    break
                d = _pairwise_dists(pts_pert)
                np.fill_diagonal(d, 0.0)
                min_pair = np.min(d) / 2.0
                min_boundary = np.min(0.5 - np.abs(pts_pert))
                cur_r = min(min_pair, min_boundary)
                
                # Gradient step to maximize min distance
                grad = np.zeros_like(pts_pert)
                for i in range(N):
                    for j in range(i + 1, N):
                        if d[i, j] < min_pair + 0.01 and d[i, j] > 1e-12:
                            diff = pts_pert[i] - pts_pert[j]
                            w = 1.0 / max(d[i, j], 1e-6)
                            grad[i] += w * diff
                            grad[j] -= w * diff
                    for k in range(2):
                        if pts_pert[i, k] > 0.5 - min_boundary:
                            grad[i, k] += 1.0
                        elif pts_pert[i, k] < -0.5 + min_boundary:
                            grad[i, k] -= 1.0
                
                pts_pert = pts_pert + step_size * grad
                pts_pert = np.clip(pts_pert, -0.49, 0.49)
            
            pts_pert_r = _compute_radius(pts_pert)
            if pts_pert_r > best_r:
                best_r = pts_pert_r
                best_pts = pts_pert.copy()
    
    # Final validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 2):
        return pts_init
    
    return best_pts

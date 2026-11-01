"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.spatial import cKDTree
import time


def _score(pts):
    """Compute the common radius for a set of points inside [-0.5, 0.5]^2."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Compute pairwise distances
    tree = cKDTree(pts)
    # Use query_pairs for efficiency
    d, i, j = tree.query_pairs(2.0, output_type='ndarray')
    if len(d) == 0:
        min_pair = np.inf
    else:
        min_pair = np.min(d)
    # Distance to boundary
    dist_to_boundary = np.min(np.min(0.5 - np.abs(pts), axis=1), axis=0)
    r = min(min_pair / 2.0, dist_to_boundary)
    return float(r)


def _soft_min_logsumexp(values, beta=100.0):
    """Approximate soft minimum using log-sum-exp trick on negative values."""
    # soft min(x) ≈ -logsumexp(-x) / beta for large beta
    neg = -values * beta
    max_neg = np.max(neg)
    if max_neg > 700:
        return -np.min(values)
    logsum = max_neg + np.log(np.sum(np.exp(neg - max_neg)))
    return -logsum / beta


def _objective_and_grad(x, beta=50.0):
    """
    Objective: maximize r = min(pairwise_dist/2, dist_to_boundary)
    We minimize -r using a smooth surrogate.
    
    x: flat array of shape (2*N,)
    Returns: (objective, gradient)
    """
    pts = x.reshape(-1, 2)
    
    # Compute all pairwise distances
    n = len(pts)
    # Use vectorized computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise radii candidates
    pair_r = dists / 2.0
    
    # Boundary distances
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    boundary_r = np.min(np.column_stack([bx, by]), axis=1)
    
    # Combine all constraints
    all_r = np.concatenate([pair_r.ravel(), boundary_r])
    
    # Soft minimum
    obj = -_soft_min_logsumexp(all_r, beta)
    
    # Gradient computation (numerical for simplicity, but let's do analytic where possible)
    # For efficiency, use finite differences on the soft min
    grad = np.zeros_like(x)
    eps = 1e-7
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = _objective_and_grad_val(xp, beta)
        fm = _objective_and_grad_val(xm, beta)
        grad[i] = (fp - fm) / (2 * eps)
    
    return obj, grad


def _objective_and_grad_val(x, beta=50.0):
    """Just the objective value without gradient."""
    pts = x.reshape(-1, 2)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_r = dists / 2.0
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    boundary_r = np.min(np.column_stack([bx, by]), axis=1)
    all_r = np.concatenate([pair_r.ravel(), boundary_r])
    return -_soft_min_logsumexp(all_r, beta)


def _hex_init():
    """Initialize with a hexagonal close-packed pattern."""
    pts = []
    # Hexagonal packing in a unit square
    # Start with a hex grid
    # For hexagonal packing, spacing = 2*r
    # Let's start with a reasonable estimate
    # For 200 circles, roughly sqrt(200) ~ 14 per side
    # Hex grid: rows offset by half spacing
    
    # Let's use a rectangular grid first, then relax
    m = int(np.ceil(np.sqrt(N)))
    # Create hexagonal pattern
    row_height = 0.866  # sqrt(3)/2
    spacing = 1.0 / m
    
    for row in range(m + 2):
        y = -0.5 + row * spacing * row_height
        if y > 0.5:
            break
        offset = 0 if row % 2 == 0 else spacing / 2
        for col in range(m + 1):
            x = -0.5 + col * spacing + offset
            if x <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts[:N])
    # Center and scale to fit
    if len(pts) > 0:
        center = pts.mean(axis=0)
        pts = pts - center
        # Scale to fit in [-0.5, 0.5]
        max_extent = np.max(np.abs(pts))
        if max_extent > 0:
            pts = pts * (0.45 / max_extent)
        pts = pts + np.array([0, 0])
    
    return pts


def _force_relaxation(pts, max_iters=200, dt=0.01):
    """Simple force-directed relaxation to increase minimum distance."""
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        forces = np.zeros_like(pts)
        
        # Repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.array([1e-10, 0])
                    dist = 1e-10
                # Repulsion stronger when close
                f = d / (dist ** 3) * 0.01
                forces[i] += f
                forces[j] -= f
        
        # Attractive force to keep in bounds
        for i in range(n):
            for k in range(2):
                if pts[i, k] > 0.4:
                    forces[i, k] -= (pts[i, k] - 0.4) * 0.5
                elif pts[i, k] < -0.4:
                    forces[i, k] -= (pts[i, k] + 0.4) * 0.5
        
        pts += dt * forces
        # Clip to bounds
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts


def _lbfgs_polish(pts, max_time=30):
    """Polish using L-BFGS-B with the smooth surrogate."""
    start_time = time.time()
    x0 = pts.ravel()
    
    # Use a moderate beta for the soft min
    beta = 50.0
    
    def obj(x):
        return _objective_and_grad_val(x, beta)
    
    # Use numerical gradient for L-BFGS-B (more robust)
    res = minimize(obj, x0, method='L-BFGS-B', 
                   options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500})
    
    pts_out = res.x.reshape(-1, 2)
    pts_out = np.clip(pts_out, -0.5, 0.5)
    
    return pts_out


def _slsqp_polish(pts, max_time=20):
    """Polish using SLSQP with explicit constraints."""
    start_time = time.time()
    x0 = pts.ravel()
    n = len(pts)
    
    def objective(x):
        # Maximize r, so minimize -r
        pts_local = x.reshape(-1, 2)
        tree = cKDTree(pts_local)
        d, i, j = tree.query_pairs(2.0, output_type='ndarray')
        if len(d) == 0:
            min_pair = np.inf
        else:
            min_pair = np.min(d)
        dist_to_boundary = np.min(np.min(0.5 - np.abs(pts_local), axis=1), axis=0)
        r = min(min_pair / 2.0, dist_to_boundary)
        return -r
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    # Constraints: all pairwise distances >= 2*r and boundary distance >= r
    # This is tricky with SLSQP, so we just use bounds and rely on the objective
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 100, 'ftol': 1e-12})
    
    pts_out = res.x.reshape(-1, 2)
    pts_out = np.clip(pts_out, -0.5, 0.5)
    
    return pts_out


def _critical_polish(pts, deadline):
    """Final high-precision polish with multiple methods."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Try L-BFGS-B polish
    if time.time() < deadline - 5:
        polished = _lbfgs_polish(best_pts)
        s = _score(polished)
        if s > best_score:
            best_pts = polished
            best_score = s
    
    # Try SLSQP polish
    if time.time() < deadline - 3:
        polished = _slsqp_polish(best_pts)
        s = _score(polished)
        if s > best_score:
            best_pts = polished
            best_score = s
    
    # Final local search: small perturbations
    if time.time() < deadline - 1:
        for _ in range(50):
            if time.time() >= deadline - 1:
                break
            perturbed = best_pts + np.random.normal(0, 0.001, size=best_pts.shape)
            perturbed = np.clip(perturbed, -0.5, 0.5)
            s = _score(perturbed)
            if s > best_score:
                best_pts = perturbed
                best_score = s
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with hexagonal initialization
    pts = _hex_init()
    
    # Force relaxation
    pts = _force_relaxation(pts, max_iters=100)
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # L-BFGS-B polish
    if time.time() < deadline - 30:
        polished = _lbfgs_polish(best_pts)
        s = _score(polished)
        if s > best_score:
            best_pts = polished
            best_score = s
    
    # SLSQP polish
    if time.time() < deadline - 15:
        polished = _slsqp_polish(best_pts)
        s = _score(polished)
        if s > best_score:
            best_pts = polished
            best_score = s
    
    # Critical polish
    if time.time() < deadline - 5:
        best_pts = _critical_polish(best_pts, deadline)
    
    # Final validation
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END

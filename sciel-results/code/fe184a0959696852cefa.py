"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import Delaunay, cKDTree


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    if len(pts) < 2:
        return 0.0
    tree = cKDTree(pts)
    min_pair_dist = tree.query_pairs(2.0)
    if len(min_pair_dist) == 0:
        min_pair_dist = np.inf
    else:
        # Compute actual distances for pairs within threshold
        pairs = list(min_pair_dist)
        diffs = pts[[p[0] for p in pairs]] - pts[[p[1] for p in pairs]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _compute_radius_fast(pts):
    """Vectorized radius computation."""
    if len(pts) < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _hex_init(n_points=300):
    """Generate hexagonal lattice points within unit disk, select n_points."""
    # Try multiple spacings
    best_pts = None
    best_r = -1.0
    for spacing in np.linspace(0.085, 0.135, 40):
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    points.append((x, y))
        if len(points) < n_points:
            continue
        points = np.array(points)
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')[:n_points]
        candidate = points[idx]
        r = _compute_radius_fast(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    return best_pts, best_r


def _soft_min_objective_and_grad(x, eps, n):
    """
    Soft-min surrogate: minimize L = sum(exp(-eps * d_ij)) + sum(exp(-eps * b_i))
    where d_ij are pairwise distances, b_i are boundary distances.
    Returns (value, gradient).
    """
    pts = x.reshape(-1, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    radii = np.hypot(pts[:, 0], pts[:, 1])
    b_dists = 1.0 - radii
    
    # Objective
    L = np.sum(np.exp(-eps * dists)) + np.sum(np.exp(-eps * b_dists))
    
    # Gradient
    weights_pair = np.exp(-eps * dists)
    np.fill_diagonal(weights_pair, 0.0)
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    grad_pair = -eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)
    
    weights_bound = np.exp(-eps * b_dists)
    safe_norms = np.where(radii > 1e-12, radii, 1.0)
    unit_radii = pts / safe_norms[:, np.newaxis]
    grad_bound = eps * weights_bound[:, np.newaxis] * unit_radii
    
    grad = (grad_pair + grad_bound).flatten()
    return L, grad


def _gradient_optimize(pts, deadline, eps_start=50.0, eps_end=500.0,
                       lr_start=0.002, lr_end=1e-5, max_iter=5000):
    """Gradient ascent on soft-min surrogate."""
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        progress = it / max_iter
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        x = best_pts.flatten()
        L, grad = _soft_min_objective_and_grad(x, eps, n)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        candidate = (best_pts.flatten() + lr * grad).reshape(-1, 2)
        
        # Enforce boundary
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            candidate[too_far] *= 0.9999 / norms_cand[too_far]
        
        cand_r = _compute_radius_fast(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = candidate.copy()
    
    return best_pts, best_r


def _get_active_edges(pts, r_target=None):
    """Get Delaunay edges that are 'active' (close to the minimum distance)."""
    if r_target is None:
        r_target = _compute_radius_fast(pts)
    min_dist = 2.0 * r_target
    # Use a threshold slightly above the minimum
    threshold = min_dist * 1.1
    tree = cKDTree(pts)
    pairs = tree.query_pairs(threshold)
    return list(pairs)


def _local_refine(pts, deadline, max_iter=2000):
    """
    Local refinement using only active edges (Delaunay neighbors).
    This is much faster than full pairwise computation.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Get active edges
        r_target = _compute_radius_fast(best_pts)
        edges = _get_active_edges(best_pts, r_target)
        
        if len(edges) == 0:
            break
        
        # Build gradient from active edges only
        grad = np.zeros_like(best_pts)
        for i, j in edges:
            diff = best_pts[i] - best_pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            # Repulsion force
            force = -diff / dist
            grad[i] += force
            grad[j] -= force
        
        # Boundary forces
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        b_dists = 1.0 - radii
        for i in range(n):
            if b_dists[i] < 0.05:
                safe_norm = max(radii[i], 1e-12)
                grad[i] += best_pts[i] / safe_norm
        
        # Normalize and apply
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad /= grad_norm
        
        lr = 0.001 * (0.99 ** (it // 100))
        candidate = best_pts + lr * grad
        
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            candidate[too_far] *= 0.9999 / norms_cand[too_far]
        
        cand_r = _compute_radius_fast(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = candidate.copy()
    
    return best_pts, best_r


def _slsqp_refine(pts, deadline):
    """
    Use SLSQP to refine the packing by maximizing the minimum distance.
    We formulate this as: maximize t subject to d_ij >= t for all pairs, |p_i| + t <= 1.
    """
    n = len(pts)
    x0 = pts.flatten()
    
    def objective(x):
        """Negative of the minimum distance (to maximize)."""
        pts = x.reshape(-1, 2)
        return -_compute_radius_fast(pts)
    
    # Constraints: all pairwise distances >= 2*r, boundary: |p_i| <= 1 - r
    # For SLSQP, we use inequality constraints: g(x) >= 0
    # We'll use a simpler approach: just maximize with boundary constraint
    
    # Boundary constraint: |p_i| <= 1 - 0.053 (approximate)
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    # Try SLSQP with a time limit
    try:
        result = opt.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-12}
        )
        if result.success or result.fun < objective(x0):
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius_fast(new_pts)
            old_r = _compute_radius_fast(pts)
            if new_r > old_r:
                return new_pts, new_r
    except Exception:
        pass
    
    return pts, _compute_radius_fast(pts)


def _polish(pts, deadline):
    """High-precision polish with very small steps."""
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    
    for lr in [5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]:
        if time.time() > deadline:
            break
        
        for it in range(500):
            if time.time() > deadline:
                break
            
            # Use local refine with active edges
            r_target = _compute_radius_fast(best_pts)
            edges = _get_active_edges(best_pts, r_target)
            
            if len(edges) == 0:
                break
            
            grad = np.zeros_like(best_pts)
            for i, j in edges:
                diff = best_pts[i] - best_pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                force = -diff / dist
                grad[i] += force
                grad[j] -= force
            
            radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
            for i in range(len(best_pts)):
                if radii[i] > 0.95:
                    grad[i] += best_pts[i] / max(radii[i], 1e-12)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            grad /= grad_norm
            
            candidate = best_pts + lr * grad
            norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
            too_far = norms_cand > 0.99999
            if np.any(too_far):
                candidate[too_far] *= 0.99999 / norms_cand[too_far]
            
            cand_r = _compute_radius_fast(candidate)
            if cand_r > best_r + 1e-15:
                best_r = cand_r
                best_pts = candidate.copy()
    
    return best_pts, best_r


def _random_perturb(pts, magnitude=0.005):
    """Add small random perturbation."""
    rng = np.random.default_rng()
    perturbed = pts + rng.normal(0, magnitude, size=pts.shape)
    norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
    too_far = norms > 0.99
    if np.any(too_far):
        perturbed[too_far] *= 0.99 / norms[too_far]
    return perturbed


def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    # Phase 0: Hexagonal initialization
    best_pts, best_r = _hex_init(N)
    
    # Phase 1: Gradient optimization with soft-min
    if time.time() < deadline - 60.0:
        best_pts, best_r = _gradient_optimize(
            best_pts, deadline - 60.0,
            eps_start=50.0, eps_end=500.0,
            lr_start=0.002, lr_end=1e-5,
            max_iter=4000
        )
    
    # Phase 2: Local refinement using active edges
    if time.time() < deadline - 30.0:
        best_pts, best_r = _local_refine(best_pts, deadline - 30.0, max_iter=3000)
    
    # Phase 3: SLSQP refinement
    if time.time() < deadline - 15.0:
        slq_pts, slq_r = _slsqp_refine(best_pts, deadline - 15.0)
        if slq_r > best_r:
            best_pts = slq_pts
            best_r = slq_r
    
    # Phase 4: Random perturbations + re-optimize
    if time.time() < deadline - 10.0:
        for attempt in range(3):
            if time.time() > deadline - 10.0:
                break
            perturbed = _random_perturb(best_pts, magnitude=0.003)
            p_pts, p_r = _local_refine(perturbed, deadline - 10.0, max_iter=1000)
            if p_r > best_r:
                best_pts = p_pts
                best_r = p_r
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 1.0:
        best_pts, best_r = _polish(best_pts, deadline - 1.0)
    
    return best_pts
# EVOLVE-BLOCK-END

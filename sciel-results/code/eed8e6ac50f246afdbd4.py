"""Pack 50 equal circles in the unit disk via ring-based parametric optimization + force-directed refinement."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_container, min_pair)


def _force_directed(P, r, n_iters=500, lr_init=0.02, lr_min=0.0001):
    """
    Force-directed layout: minimize overlaps using pairwise repulsion and boundary attraction.
    """
    P = P.copy()
    n = len(P)
    lr = lr_init
    decay = (lr_min / lr_init) ** (1.0 / max(n_iters - 1, 1))
    
    for it in range(n_iters):
        # Compute forces
        forces = np.zeros_like(P)
        
        # Pairwise repulsion (only for pairs that are too close)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Repulsive force: push apart if dist < 2r
        # Use a smooth approximation
        overlap = np.maximum(2.0 * r - dist, 0.0)
        inv_dist = np.zeros_like(dist)
        np.divide(1.0, dist, out=inv_dist, where=dist > 1e-10)
        
        # Force magnitude proportional to overlap
        force_mag = overlap * inv_dist
        # Sum forces from all other particles
        f_pair = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        forces += f_pair
        
        # Boundary repulsion: push inward if dist_from_center + r > 1
        dist_origin = np.linalg.norm(P, axis=1)
        boundary_overlap = np.maximum(dist_origin + r - 1.0, 0.0)
        inv_dist_origin = np.zeros_like(dist_origin)
        np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 1e-10)
        
        # Force pushing toward center
        f_boundary = boundary_overlap * inv_dist_origin[:, np.newaxis] * P
        forces -= f_boundary
        
        # Update positions with adaptive learning rate
        P += lr * forces
        lr *= decay
        
        # Constrain to unit circle (soft projection)
        dist_origin = np.linalg.norm(P, axis=1)
        mask = dist_origin > 1.0 - r
        if np.any(mask):
            scale = (1.0 - r) / dist_origin[mask]
            P[mask] *= scale[:, np.newaxis]
    
    return P


def _get_ring_initialization(params, ring_counts):
    """Generate points based on concentric rings."""
    P = np.zeros((N, 2))
    k = len(ring_counts)
    radii = params[:k]
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
    return P


def _optimize_positions(P_init, r_target, time_limit=5.0):
    """Optimize positions using L-BFGS-B on smooth energy."""
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        overlaps_cc = np.maximum(2.0 * r_target - dist, 0.0)
        dist_origin = np.linalg.norm(P, axis=1)
        overlaps_cb = np.maximum(dist_origin + r_target - 1.0, 0.0)
        return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        grad = np.zeros_like(P)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        V_cc = np.maximum(2.0 * r_target - dist, 0.0)
        inv_dist = np.zeros_like(dist)
        np.divide(1.0, dist, out=inv_dist, where=dist > 0)
        W = V_cc * inv_dist
        sum_W = np.sum(W, axis=1)
        term1 = (P.T @ W).T
        term2 = P * sum_W[:, np.newaxis]
        grad += term1 - term2
        dist_origin = np.linalg.norm(P, axis=1)
        V_cb = np.maximum(r_target + dist_origin - 1.0, 0.0)
        inv_dist_origin = np.zeros_like(dist_origin)
        np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
        grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
        return grad.flatten()
    
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())


def _global_search_ring_params(ring_counts, time_limit=20.0):
    """Use differential evolution to find good ring radii and offsets."""
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        P = _get_ring_initialization(params, ring_counts)
        score = _compute_score(P)
        return -score
    
    try:
        res = differential_evolution(neg_score, bounds, maxiter=40, popsize=12,
                                     tol=1e-5, seed=42, polish=True, workers=1,
                                     init='latinhypercube')
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
    except Exception:
        best_P = _get_hex_initial_points()
    return best_P


def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16]
    ]
    
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    for ring_counts in ring_structures:
        if time.time() > deadline - 40.0:
            break
        
        # 1. Global search for ring parameters
        P_candidate = _global_search_ring_params(ring_counts, time_limit=20.0)
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Force-directed refinement to escape local minima
        r_fd = score_best * 1.01
        P_fd = _force_directed(P_candidate, r_fd, n_iters=300, lr_init=0.015, lr_min=0.00005)
        score_fd = _compute_score(P_fd)
        if score_fd > score_best:
            P_best = P_fd
            score_best = score_fd
        
        # 3. Binary search on r with local refinement
        r_low = max(score_best * 0.95, 0.12)
        r_high = 0.145
        
        # Feasibility check at r_low
        P_opt, E = _optimize_positions(P_best.copy(), r_low, time_limit=4.0)
        if E < 1e-8:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
            r_low = score
        
        for _ in range(10):
            if time.time() > deadline - 12.0:
                break
            
            r_mid = (r_low + r_high) / 2.0
            P_init = P_best.copy()
            
            # Use force-directed first to escape potential local minima
            P_fd = _force_directed(P_init, r_mid, n_iters=150, lr_init=0.01, lr_min=0.0001)
            P_opt, E = _optimize_positions(P_fd, r_mid, time_limit=4.0)
            
            if E < 1e-8:
                score = _compute_score(P_opt)
                if score > score_best:
                    P_best = P_opt
                    score_best = score
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        r_target = score_best * 1.0005
        P_fd = _force_directed(P_best.copy(), r_target, n_iters=200, lr_init=0.005, lr_min=0.00001)
        P_opt, E = _optimize_positions(P_fd, r_target, time_limit=5.0)
        if E < 1e-10:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt
                score_best = score
    
    # Validate
    radii = np.sqrt(np.sum(P_best ** 2, axis=1))
    if np.any(~np.isfinite(P_best)) or np.any(radii >= 1.0):
        P_best = _get_hex_initial_points()
    
    return P_best

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, Bounds

N = 50
SEED_RADIUS = 0.1258254895304  # approximate best known

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
import time


def _score(pts):
    """Compute the packing radius: min of half min-pair-distance and min margin to boundary."""
    pts = np.asarray(pts)
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundary
    r = np.linalg.norm(pts, axis=1)
    min_margin = 1.0 - np.max(r)
    return min(min_pair, min_margin)


def _soft_score(pts, beta=500.0):
    """Smooth surrogate using log-sum-exp for the min."""
    pts = np.asarray(pts)
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    # soft min of pair distances / 2
    m = np.min(dists)
    log_sum_exp = m + np.log(np.sum(np.exp((dists - m) / beta)))
    soft_pair = log_sum_exp / 2.0
    # soft max of radii
    r = np.linalg.norm(pts, axis=1)
    m2 = np.max(r)
    log_sum_exp2 = m2 + np.log(np.sum(np.exp((r - m2) / beta)))
    soft_margin = 1.0 - log_sum_exp2
    return min(soft_pair, soft_margin)


def _init_concentric_rings():
    """
    Initialize with concentric ring structure.
    Known good packing for 50 circles uses rings like:
    center (1), ring1 (6), ring2 (12), ring3 (18), ring4 (13) or similar.
    Let's use a parametric ring structure.
    """
    # Try a few different ring configurations
    configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 18, 12, 1],
        [1, 6, 14, 16, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 14, 16, 13],
    ]
    
    best_pts = None
    best_score = -1
    
    for config in configs:
        if sum(config) != N:
            continue
        
        pts_list = []
        # Center
        pts_list.append([0.0, 0.0])
        
        ring_radius = 0.25  # initial guess
        for ring_idx, count in enumerate(config[1:]):
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for a in angles:
                pts_list.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
            ring_radius += 0.22
        
        pts = np.array(pts_list)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    if best_pts is None:
        # Fallback: hexagonal grid
        best_pts = _init_hex()
    
    return best_pts


def _init_hex():
    """Hexagonal lattice initialization."""
    pts_list = []
    spacing = 0.25
    k = int(1.0 / spacing) + 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                pts_list.append([x, y])
    pts = np.array(pts_list)
    if len(pts) > N:
        r = np.linalg.norm(pts, axis=1)
        idx = np.argsort(r, kind='stable')[:N]
        pts = pts[idx]
    return pts


def _optimize_ring_structure():
    """
    Optimize a parametric ring structure using L-BFGS-B.
    Parameters: ring radii and angular offsets.
    """
    deadline = time.time() + 100  # leave safety margin
    
    # Use the known ring structure: 1, 6, 12, 18, 13
    ring_counts = [1, 6, 12, 18, 13]
    
    def build_points(params):
        pts_list = []
        # Center
        pts_list.append([0.0, 0.0])
        
        for ring_idx in range(1, len(ring_counts)):
            count = ring_counts[ring_idx]
            # Parameters: radius, angle_offset, and possibly per-circle angular deviations
            # Keep it simple: radius and angle offset per ring
            radius = params[ring_idx * 2]
            angle_offset = params[ring_idx * 2 + 1]
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + angle_offset
            for a in angles:
                pts_list.append([radius * np.cos(a), radius * np.sin(a)])
        
        return np.array(pts_list)
    
    def negative_score(params):
        if time.time() > deadline:
            return 1e10
        pts = build_points(params)
        if np.any(~np.isfinite(pts)):
            return 1e10
        return -_score(pts)
    
    # Initial parameters: ring radii and angle offsets
    init_params = []
    ring_radius = 0.25
    for i in range(1, len(ring_counts)):
        init_params.append(ring_radius)
        init_params.append(0.0)
        ring_radius += 0.22
    
    init_params = np.array(init_params)
    
    # Bounds: radii between 0 and 1, angles between -pi and pi
    bounds = []
    for i in range(1, len(ring_counts)):
        bounds.append((0.0, 1.0))  # radius
        bounds.append((-np.pi, np.pi))  # angle offset
    
    result = minimize(negative_score, init_params, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
    
    best_pts = build_points(result.x)
    best_s = _score(best_pts)
    
    # Local refinement: optimize individual circle positions
    if time.time() < deadline - 10:
        best_pts, best_s = _local_refine(best_pts, best_s, deadline)
    
    return best_pts, best_s


def _local_refine(pts, current_score, deadline):
    """
    Local refinement by optimizing individual circle positions.
    Use L-BFGS-B on all 2*N coordinates.
    """
    def negative_score_all(x):
        if time.time() > deadline:
            return 1e10
        p = x.reshape(-1, 2)
        if np.any(~np.isfinite(p)):
            return 1e10
        return -_score(p)
    
    x0 = pts.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    result = minimize(negative_score_all, x0, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
    
    new_pts = result.x.reshape(-1, 2)
    new_score = _score(new_pts)
    
    if new_score > current_score:
        return new_pts, new_score
    return pts, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin for 120s limit
    
    best_pts = _init_concentric_rings()
    best_score = _score(best_pts)
    
    # Optimize ring structure
    if time.time() < deadline - 20:
        pts2, score2 = _optimize_ring_structure()
        if score2 > best_score:
            best_pts = pts2
            best_score = score2
    
    # Additional local refinement with different initializations
    if time.time() < deadline - 5:
        # Try hex initialization too
        hex_pts = _init_hex()
        if len(hex_pts) == N:
            hex_pts_r, hex_score = _local_refine(hex_pts, _score(hex_pts), deadline)
            if hex_score > best_score:
                best_pts = hex_pts_r
                best_score = hex_score
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        def negative_score_polish(x):
            if time.time() > deadline:
                return 1e10
            p = x.reshape(-1, 2)
            if np.any(~np.isfinite(p)):
                return 1e10
            return -_score(p)
        
        x0 = best_pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        result = minimize(negative_score_polish, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        
        final_pts = result.x.reshape(-1, 2)
        final_score = _score(final_pts)
        
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    # Ensure validity
    if not np.all(np.isfinite(best_pts)):
        best_pts = _init_hex()
    
    # Clip to ensure inside unit circle
    r = np.linalg.norm(best_pts, axis=1)
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r[mask, np.newaxis] * 0.999
    
    return best_pts
# EVOLVE-BLOCK-END

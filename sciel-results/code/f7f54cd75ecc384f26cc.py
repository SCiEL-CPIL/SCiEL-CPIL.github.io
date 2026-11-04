"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree

def _hex_init():
    """Generate hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.97

def _compute_radius(pts):
    """Compute the packing radius."""
    if len(pts) != N:
        return 0.0
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(boundary))

def _soft_min_value(pts, beta=50.0):
    """Soft-min surrogate for the packing radius using log-sum-exp."""
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Soft min of boundary and pair distances
    all_vals = np.concatenate([boundary, pair_dists[np.triu_indices(N, k=1)]])
    # Clip to avoid overflow
    all_vals = np.clip(all_vals, 1e-10, 10.0)
    
    m = np.min(all_vals)
    soft_min = m - np.log(np.sum(np.exp(-beta * (all_vals - m)))) / beta
    return soft_min

def _force_relax(pts, r_target, iters=200, lr=0.01, deadline=None):
    """Vectorized force-directed relaxation."""
    pts = pts.copy()
    for it in range(iters):
        if deadline and time.time() > deadline:
            break
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Only consider pairs that are too close
        mask = dists < 2.0 * r_target
        if not np.any(mask):
            break
        
        # Compute forces vectorized
        forces = np.zeros_like(pts)
        # For each close pair, add repulsive force
        for i in range(N):
            close = np.where(mask[i])[0]
            if len(close) == 0:
                continue
            d = pts[i] - pts[close]
            norms = dists[i, close]
            norms = np.maximum(norms, 1e-10)
            f = (2.0 * r_target - norms)[:, np.newaxis] * d / norms[:, np.newaxis]
            forces[i] += np.sum(f, axis=0)
            for j_idx, j in enumerate(close):
                forces[j] -= f[j_idx]
        
        # Boundary forces
        radii = np.linalg.norm(pts, axis=1)
        over = radii > (1.0 - r_target)
        if np.any(over):
            dir_vec = pts[over] / np.maximum(radii[over], 1e-10)[:, np.newaxis]
            forces[over] -= (radii[over] - (1.0 - r_target))[:, np.newaxis] * dir_vec
        
        # Apply with adaptive learning rate
        pts += lr * forces
        
        # Project back inside unit circle
        new_radii = np.linalg.norm(pts, axis=1)
        outside = new_radii > 1.0
        if np.any(outside):
            pts[outside] = pts[outside] / new_radii[outside][:, np.newaxis]
    
    return pts

def _get_neighbors(pts, k=12):
    """Get k nearest neighbors for each point."""
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k+1)
    return indices

def _local_refine(pts, deadline):
    """Iterative local refinement with force relaxation and neighbor-based swaps."""
    pts = pts.copy()
    current_score = _compute_radius(pts)
    
    # Get neighbor graph
    neighbors = _get_neighbors(pts, k=10)
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        r_target = current_score * 1.001
        relaxed = _force_relax(pts, r_target, iters=100, lr=0.005, deadline=deadline)
        new_score = _compute_radius(relaxed)
        
        if new_score > current_score + 1e-12:
            pts = relaxed
            current_score = new_score
            # Update neighbors periodically
            if iteration % 5 == 0:
                neighbors = _get_neighbors(pts, k=10)
        else:
            # Try small perturbations of the worst points
            break
    
    return pts, current_score

def _anneal_restart(pts, deadline, n_restarts=3):
    """Simulated annealing-style restarts to escape local minima."""
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb worst points
        pts_copy = best_pts.copy()
        diff = pts_copy[:, np.newaxis, :] - pts_copy[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find the worst (closest) pairs
        min_dists = np.min(dists, axis=1)
        worst_points = np.argsort(min_dists)[:30]
        
        # Perturb them
        noise = np.random.normal(0, 0.01, size=(len(worst_points), 2))
        pts_copy[worst_points] += noise
        
        # Relax
        r_target = best_score * 0.95
        relaxed = _force_relax(pts_copy, r_target, iters=150, lr=0.005, deadline=deadline)
        score = _compute_radius(relaxed)
        
        if score > best_score:
            best_pts = relaxed
            best_score = score
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 125  # Leave 25s buffer for safety
    
    # Initialize with hexagonal lattice
    best_pts = _hex_init()
    best_score = _compute_radius(best_pts)
    
    # Phase 1: Force relaxation
    if time.time() < deadline:
        r_target = best_score * 1.01
        relaxed = _force_relax(best_pts, r_target, iters=300, lr=0.005, deadline=deadline)
        score = _compute_radius(relaxed)
        if score > best_score:
            best_pts = relaxed
            best_score = score
    
    # Phase 2: Local refinement
    if time.time() < deadline:
        refined, refined_score = _local_refine(best_pts, deadline)
        if refined_score > best_score:
            best_pts = refined
            best_score = refined_score
    
    # Phase 3: Annealing restarts
    if time.time() < deadline:
        annealed, annealed_score = _anneal_restart(best_pts, deadline, n_restarts=3)
        if annealed_score > best_score:
            best_pts = annealed
            best_score = annealed_score
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline:
        r_target = best_score * 1.0005
        polished = _force_relax(best_pts, r_target, iters=500, lr=0.001, deadline=deadline)
        polished_score = _compute_radius(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Final validation and cleanup
    radii = np.linalg.norm(best_pts, axis=1)
    if np.any(radii > 1.0):
        best_pts = best_pts / radii[:, np.newaxis]
    
    # Ensure exactly N points
    if len(best_pts) != N:
        # Fallback
        best_pts = _hex_init()
    
    return best_pts
# EVOLVE-BLOCK-END

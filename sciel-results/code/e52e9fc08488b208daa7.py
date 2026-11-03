"""Improved packing: hex lattice init + iterative local optimization with time guard."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time


def _score(pts):
    """Compute the packing radius (min of half min-pair-dist and min boundary margin)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    tree = cKDTree(pts)
    # Use query_pairs for efficiency
    pairs = tree.query_pairs(r=0.1)  # upper bound on distance to check
    if len(pairs) == 0:
        min_pair = np.inf
    else:
        # Compute distances for all pairs
        d2 = np.array([np.sum((pts[i] - pts[j]) ** 2) for i, j in pairs])
        min_pair = np.sqrt(np.min(d2)) / 2.0 if len(d2) > 0 else np.inf
    # Boundary
    norms = np.linalg.norm(pts, axis=1)
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair, min_boundary)


def _init_hex():
    """Generate a hexagonal lattice point set and select N points inside unit disk."""
    # Target radius ~ 0.053, so spacing ~ 0.106
    spacing = 0.106
    while True:
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            # Sort by distance from origin, take innermost N
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii, kind="stable")[:N]
            return pts[idx].copy()
        spacing *= 0.995


def _optimize(pts, time_limit=140.0):
    """Iterative repulsion-based local optimization."""
    start = time.time()
    r = _score(pts)
    n = len(pts)
    
    # Parameters
    learning_rate = 0.005
    decay = 0.999
    min_lr = 1e-6
    
    for it in range(20000):
        if time.time() - start > time_limit:
            break
        
        # Compute forces: repulsion from all other points + attraction to center for boundary
        # We want to maximize min distance, so push apart points that are too close
        tree = cKDTree(pts)
        # For each point, find nearest neighbors
        dists, idxs = tree.query(pts, k=6)  # 6 nearest neighbors (hex coordination)
        
        forces = np.zeros_like(pts)
        for i in range(n):
            for k in range(1, 6):  # skip self
                j = idxs[i, k]
                d = dists[i, k]
                if d < 1e-10:
                    # Coincident: random push
                    forces[i] += np.random.randn(2) * 0.01
                    continue
                # Repulsive force, stronger when closer
                diff = pts[i] - pts[j]
                # Force magnitude ~ 1/d^2 for strong repulsion at close range
                f_mag = 1.0 / (d * d + 1e-8)
                forces[i] += f_mag * diff / d
        
        # Boundary constraint: pull inward if too close to boundary
        norms = np.linalg.norm(pts, axis=1)
        boundary_margin = 1.0 - norms
        for i in range(n):
            if boundary_margin[i] < 0.05:
                # Pull toward center
                forces[i] -= pts[i] / (norms[i] + 1e-10) * 0.5
        
        # Normalize forces to prevent explosion
        force_norms = np.linalg.norm(forces, axis=1)
        max_fn = np.max(force_norms)
        if max_fn > 0:
            forces = forces / max_fn * 0.01
        
        # Update
        pts = pts + learning_rate * forces
        
        # Enforce boundary strictly
        norms = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if norms[i] > 0.999:
                pts[i] = pts[i] / norms[i] * 0.999
        
        # Update learning rate
        learning_rate *= decay
        if learning_rate < min_lr:
            break
        
        # Compute score periodically
        if it % 10 == 0:
            new_r = _score(pts)
            if new_r > r:
                r = new_r
                best_pts = pts.copy()
            # If score dropped significantly, revert
            elif new_r < r * 0.95:
                pts = best_pts.copy()
                learning_rate *= 0.9
        
    return best_pts if 'best_pts' in dir() else pts, r


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    
    # Initialize
    pts = _init_hex()
    best_pts = pts.copy()
    best_r = _score(pts)
    
    # Optimization
    pts, r = _optimize(pts, time_limit=deadline - (time.time() - start_time))
    if r > best_r:
        best_r = r
        best_pts = pts
    
    # Final check
    final_r = _score(best_pts)
    if final_r > best_r:
        best_r = final_r
    
    return best_pts
# EVOLVE-BLOCK-END

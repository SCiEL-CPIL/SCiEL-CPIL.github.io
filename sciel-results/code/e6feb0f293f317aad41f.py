"""Optimized packing of 300 equal circles in a unit circle using gradient-based optimization."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hexagonal_init(n, rng):
    """Generate n points in a hexagonal lattice pattern inside the unit circle."""
    # Create a hexagonal lattice
    dx = 0.1
    dy = dx * np.sqrt(3) / 2
    points = []
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if x * x + y * y <= 0.95:
                points.append([x, y])
    points = np.array(points)
    if len(points) < n:
        # Fallback: random
        return rng.random((n, 2)) * 2 - 1
    # Shuffle and take n
    idx = rng.permutation(len(points))[:n]
    return points[idx]


def _compute_score(pts):
    """Compute the maximum radius r for the given centers."""
    # Minimum distance between any two centers / 2
    # Use scipy's cKDTree for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    # Find nearest neighbors
    dists, _ = tree.query(pts, k=2)
    min_inter = np.min(dists[:, 1]) / 2.0
    # Minimum distance to boundary
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = 1.0 - np.max(radii)
    return min(min_inter, min_boundary)


def _gradient_descent(pts, max_iters=5000, learning_rate=0.001, deadline=None):
    """Gradient-based optimization to maximize the minimum distance."""
    from scipy.spatial import cKDTree
    
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces (gradients)
        tree = cKDTree(pts)
        dists, indices = tree.query(pts, k=2)
        
        # Force from nearest neighbor
        forces = np.zeros_like(pts)
        for i in range(len(pts)):
            j = indices[i, 1]
            if j != i:
                d = pts[i] - pts[j]
                dist = dists[i, 1]
                if dist > 1e-10:
                    # Repulsive force
                    forces[i] += d / (dist * dist)
        
        # Boundary force
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_margin = 1.0 - radii
        for i in range(len(pts)):
            if boundary_margin[i] < 0.1:
                # Push inward
                if radii[i] > 1e-10:
                    forces[i] -= pts[i] / (radii[i] * radii[i])
        
        # Update positions
        pts_new = pts + learning_rate * forces
        # Keep within unit circle
        radii_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = radii_new > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / radii_new[mask]
        
        score = _compute_score(pts_new)
        if score > best_score:
            best_score = score
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # Reduce learning rate
            learning_rate *= 0.95
            if learning_rate < 1e-6:
                break
    
    return best_pts, best_score


def construct():
    """Construct 300 circle centers in the unit circle to maximize the common radius."""
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    rng = np.random.default_rng(42)
    
    # Start with hexagonal lattice
    pts = _hexagonal_init(N, rng)
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    # Multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Use the hexagonal init
            pts = _hexagonal_init(N, rng)
        else:
            # Add random perturbation
            pts = best_pts + rng.random((N, 2)) * 0.05
            # Keep within unit circle
            radii = np.hypot(pts[:, 0], pts[:, 1])
            mask = radii > 0.95
            if np.any(mask):
                pts[mask] = pts[mask] * 0.95 / radii[mask]
        
        # Run gradient descent
        pts_opt, score = _gradient_descent(pts, max_iters=2000, learning_rate=0.002, deadline=deadline)
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    return best_pts
# EVOLVE-BLOCK-END

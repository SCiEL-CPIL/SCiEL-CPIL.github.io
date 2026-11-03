"""Improved packing: ring-based init + multi-stage optimization with deadline guard."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill


def compute_score(points):
    """Compute the packing radius for given centers."""
    if len(points) != N:
        return -1.0
    # Boundary clearance
    dist_to_center = np.linalg.norm(points, axis=1)
    boundary_clearance = 1.0 - dist_to_center
    # Pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_clearance, min_pair_dist / 2.0)


def ring_init(n_circles=50):
    """Initialize with concentric rings."""
    points = []
    # Center circle
    points.append(np.array([0.0, 0.0]))
    remaining = n_circles - 1
    
    # Try different ring configurations
    # Known optimal has roughly: 1 center, 6 inner, 12 mid, 15 outer, 16 outermost
    # Let's try: 1, 6, 12, 15, 16
    ring_counts = [1, 6, 12, 15, 16]
    ring_radii = [0.0, 0.25, 0.5, 0.72, 0.92]
    
    for r_count, r_radius in zip(ring_counts, ring_radii):
        if r_count == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, r_count, endpoint=False)
        for angle in angles:
            x = r_radius * np.cos(angle)
            y = r_radius * np.sin(angle)
            points.append(np.array([x, y]))
    
    # If we have too many or too few, adjust
    while len(points) < n_circles:
        # Add more to outer ring
        outer_r = 0.95
        angle = np.random.uniform(0, 2 * np.pi)
        points.append(outer_r * np.array([np.cos(angle), np.sin(angle)]))
    while len(points) > n_circles:
        points.pop()
    
    return np.array(points[:n_circles])


def optimize_points(points, max_iter=5000, lr_schedule=None):
    """Optimize points using gradient ascent on the minimum distance."""
    points = points.copy()
    best_score = compute_score(points)
    best_points = points.copy()
    
    # Precompute which pairs are closest
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Compute all pairwise distances
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum pairwise distance
        min_idx = np.argmin(dists)
        i, j = np.unravel_index(min_idx, (N, N))
        
        # Also check boundary
        dist_to_center = np.linalg.norm(points, axis=1)
        boundary_clearance = 1.0 - dist_to_center
        min_boundary_idx = np.argmin(boundary_clearance)
        
        # Determine which constraint is binding
        min_pair_dist = dists[i, j]
        min_boundary = boundary_clearance[min_boundary_idx]
        
        # Gradient computation
        grad = np.zeros_like(points)
        
        if min_pair_dist < min_boundary:
            # Pairwise constraint is binding
            diff = points[i] - points[j]
            dist = min_pair_dist
            if dist > 1e-12:
                direction = diff / dist
                grad[i] += direction
                grad[j] -= direction
        else:
            # Boundary constraint is binding
            p = points[min_boundary_idx]
            dist = np.linalg.norm(p)
            if dist > 1e-12:
                direction = p / dist
                grad[min_boundary_idx] -= direction
        
        # Learning rate with decay
        if lr_schedule is None:
            lr = 0.001 * (1.0 / (1.0 + it * 0.001))
        else:
            lr = lr_schedule(it)
        
        # Update
        points += lr * grad
        
        # Clamp to stay within unit circle
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points[mask] = points[mask] / norms[mask, np.newaxis] * 0.999
        
        # Track best
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    return best_points, best_score


def multi_stage_optimize(initial_points, stages=[(1000, 0.005), (2000, 0.002), (3000, 0.0005)]):
    """Multi-stage optimization with decreasing learning rates."""
    points = initial_points.copy()
    best_score = compute_score(points)
    best_points = points.copy()
    
    for max_iter, base_lr in stages:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        def lr_schedule(it):
            return base_lr * (1.0 / (1.0 + it * 0.0005))
        
        points, score = optimize_points(points, max_iter=max_iter, lr_schedule=lr_schedule)
        if score > best_score:
            best_score = score
            best_points = points
    
    return best_points, best_score


def construct():
    """Main entry point: return optimal centers for 50 circles in unit disk."""
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 140.0
    
    # Try multiple initializations and pick the best
    best_points = None
    best_score = -1.0
    
    # Strategy 1: Ring-based initialization
    np.random.seed(42)
    for trial in range(5):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Vary the ring radii slightly
        points = ring_init(N)
        
        # Add small random perturbation
        points += np.random.normal(0, 0.01, size=points.shape)
        
        # Optimize
        opt_points, score = multi_stage_optimize(points)
        if score > best_score:
            best_score = score
            best_points = opt_points
    
    # Strategy 2: Hexagonal lattice as fallback
    if time.time() - START_TIME < TIME_LIMIT - 10:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        pts = pts[mask]
        if len(pts) >= N:
            hex_points = pts[np.argsort(radius, kind="stable")[:N]]
            opt_points, score = multi_stage_optimize(hex_points)
            if score > best_score:
                best_score = score
                best_points = opt_points
    
    # Final polish with very small steps
    if best_points is not None and time.time() - START_TIME < TIME_LIMIT - 5:
        def final_lr(it):
            return 0.0001 * (1.0 / (1.0 + it * 0.001))
        polished, score = optimize_points(best_points, max_iter=2000, lr_schedule=final_lr)
        if score > best_score:
            best_score = score
            best_points = polished
    
    if best_points is None:
        # Emergency fallback
        best_points = ring_init(N)
    
    return best_points

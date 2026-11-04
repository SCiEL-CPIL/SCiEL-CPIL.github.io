"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130
MAX_TIME = 110.0  # seconds, safety margin before 120s kill


def _covering_radius(points):
    """Compute the covering radius in degrees."""
    # For each point, find the nearest other point
    # The covering radius is half the max nearest-neighbor distance
    # Actually, the covering radius is the maximum over all sphere points of
    # distance to nearest config point. For a set of points on the sphere,
    # this equals half the maximum of the nearest-neighbor distances
    # (the Voronoi cell circumradius on the sphere).
    
    # Compute pairwise angular distances
    dots = np.clip(points @ points.T, -1.0, 1.0)
    angles = np.arccos(dots)
    np.fill_diagonal(angles, 0.0)
    
    # Nearest neighbor distance for each point
    nn_dists = np.min(angles, axis=1)
    
    # The covering radius is half the maximum nearest-neighbor distance
    # This is an approximation; the true covering radius is the max circumradius
    # of Voronoi cells, but for well-distributed points, half the max NN distance
    # is a good proxy.
    return np.max(nn_dists) / 2.0 * 180.0 / np.pi


def _gradient_step(points, lr, t):
    """One gradient descent step to maximize the minimum pairwise distance."""
    dots = np.clip(points @ points.T, -1.0, 1.0)
    angles = np.arccos(dots)
    np.fill_diagonal(angles, 0.0)
    
    # We want to maximize the minimum angle, equivalently minimize the max
    # of (1 - cos(angle)) for the closest pairs.
    # Use a soft-min approach: for each pair, weight by how close they are.
    
    # Gradient of the covering radius surrogate
    # Use the idea: for each point, push it away from its nearest neighbors
    
    nn_dists = np.min(angles, axis=1)
    min_dist = np.min(nn_dists)
    
    # For each point, compute the gradient that pushes it away from its nearest neighbor
    # The gradient of angle with respect to point i (moving point i) when point j is nearest:
    # d/dxi arccos(xi.xj) = -xj / sqrt(1 - (xi.xj)^2)
    # But we want to INCREASE the angle, so move xi away from xj.
    
    grad = np.zeros_like(points)
    for i in range(N):
        # Find nearest neighbor
        j = np.argmin(angles[i])
        if i == j:
            continue
        # Push point i away from point j
        # The direction to move is (xi - (xi.xj)xj) / ||xi - (xi.xj)xj||
        dot_ij = dots[i, j]
        direction = points[i] - dot_ij * points[j]
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-10:
            direction /= norm_dir
            grad[i] = direction
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-10:
        grad /= grad_norm
    
    # Take step and re-normalize
    new_points = points + lr * grad
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    new_points = new_points / norms
    
    return new_points


def construct():
    start_time = time.time()
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Multi-start with different initializations and gradient descent
    rng = np.random.default_rng(42)
    
    # Try different learning rates and step counts
    lr_schedule = [0.05, 0.03, 0.01, 0.005]
    
    for lr in lr_schedule:
        if time.time() - start_time > MAX_TIME:
            break
        
        current_points = points.copy()
        for step in range(500):
            if time.time() - start_time > MAX_TIME:
                break
            
            current_points = _gradient_step(current_points, lr, step)
            
            if step % 50 == 0:
                cr = _covering_radius(current_points)
                if cr < best_radius:
                    best_radius = cr
                    best_points = current_points.copy()
        
        # Check final
        cr = _covering_radius(current_points)
        if cr < best_radius:
            best_radius = cr
            best_points = current_points.copy()
    
    # Try a few random restarts
    for restart in range(3):
        if time.time() - start_time > MAX_TIME:
            break
        
        # Random initialization
        current_points = rng.standard_normal((N, 3))
        current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
        
        lr = 0.01
        for step in range(300):
            if time.time() - start_time > MAX_TIME:
                break
            
            current_points = _gradient_step(current_points, lr, step)
            
            if step % 50 == 0:
                cr = _covering_radius(current_points)
                if cr < best_radius:
                    best_radius = cr
                    best_points = current_points.copy()
        
        cr = _covering_radius(current_points)
        if cr < best_radius:
            best_radius = cr
            best_points = current_points.copy()
    
    # Final check
    if time.time() - start_time < MAX_TIME:
        cr = _covering_radius(best_points)
        if cr < best_radius:
            best_radius = cr
    
    # Ensure all points are on the unit sphere
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points

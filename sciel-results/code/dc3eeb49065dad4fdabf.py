import numpy as np
import time

N = 50

def _hex_lattice_init(n_points):
    """Generate n_points from a hexagonal lattice inside the unit circle."""
    # Try different spacings to get a good initial distribution
    best_points = None
    best_score = -1.0
    
    # Spacing roughly related to sqrt(area/n)
    # Area = pi. Area per point = pi/n.
    # Hex cell area approx s^2 * sqrt(3)/2.
    # s ~ sqrt(2*pi / (n*sqrt(3)))
    base_spacing = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    for spacing in np.linspace(base_spacing * 0.8, base_spacing * 1.2, 20):
        k = int(1.0 / spacing) + 2
        xs = []
        ys = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    xs.append(x)
                    ys.append(y)
        
        if len(xs) >= n_points:
            points = np.array(list(zip(xs, ys)))[:n_points]
            
            # Calculate score (min radius)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            min_dist = np.min(dist[np.triu_indices(n_points, k=1)])
            norms = np.linalg.norm(points, axis=1)
            max_norm = np.max(norms)
            score = min(min_dist / 2.0, 1.0 - max_norm)
            
            if score > best_score:
                best_score = score
                best_points = points

    if best_points is None:
        # Fallback
        best_points = np.random.uniform(-0.9, 0.9, (n_points, 2))
        
    return best_points

def _optimize(points, r, iterations=1000, lr=0.01):
    """
    Optimize points to fit radius r using gradient ascent on validity.
    """
    points = points.copy()
    n = len(points)
    
    # Pre-compute indices for upper triangle
    iu = np.triu_indices(n, k=1)
    
    for _ in range(iterations):
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist[dist == 0] = 1e-10
        
        # Overlap condition: dist < 2r
        # We want to push apart if dist < 2r
        # Force magnitude: (2r - dist)
        # Direction: diff / dist
        
        # Only compute for upper triangle to save time?
        # No, vectorized full matrix is fast for N=50.
        
        overlap = dist < 2 * r
        force_mag = np.where(overlap, (2 * r - dist) / dist, 0)
        
        # Sum forces
        # force_vecs[i] = sum_j force_mag[i,j] * diff[i,j]
        force_vecs = np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        forces += force_vecs
        
        # Boundary constraint
        norms = np.linalg.norm(points, axis=1)
        boundary_violation = norms > (1 - r)
        
        # Force magnitude: (norm - (1-r))
        # Direction: -p / norm
        bound_force_mag = np.where(boundary_violation, (norms - (1 - r)) / norms, 0)
        boundary_forces = -bound_force_mag[:, np.newaxis] * points
        
        forces += boundary_forces
        
        # Update
        points += lr * forces
        
        # Optional: Check for convergence
        # If forces are small, we might be at a local min/max
        # But we want to maximize r, so we just want to resolve overlaps.
        
    # Final check
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    min_dist = np.min(dist[iu])
    norms = np.linalg.norm(points, axis=1)
    max_norm = np.max(norms)
    
    if min_dist >= 2 * r and max_norm <= 1 - r:
        return points, True
    else:
        return points, False

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # 1. Initialization
    points = _hex_lattice_init(N)
    
    def calc_r(pts):
        n = len(pts)
        iu = np.triu_indices(n, k=1)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        min_dist = np.min(dist[iu])
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        return min(min_dist / 2.0, 1.0 - max_norm)
    
    current_r = calc_r(points)
    best_r = current_r
    best_points = points.copy()
    
    # 2. Optimization Loop
    r_step = 0.0005
    failed_count = 0
    
    while time.time() < deadline:
        target_r = best_r + r_step
        
        # Optimize
        # Use more iterations if r is large?
        opt_points, success = _optimize(best_points, target_r, iterations=500, lr=0.02)
        
        if success:
            best_r = target_r
            best_points = opt_points
            failed_count = 0
            # Increase step slightly to speed up
            r_step = min(r_step * 1.1, 0.001)
        else:
            failed_count += 1
            # Reduce step
            r_step *= 0.8
            if r_step < 1e-5:
                if failed_count > 20:
                    # Stuck. Perturb and restart
                    # Add random noise
                    noise = np.random.normal(0, 0.05, best_points.shape)
                    new_points = best_points + noise
                    # Project back to circle if too far
                    norms = np.linalg.norm(new_points, axis=1)
                    mask = norms > 1.0
                    new_points[mask] = new_points[mask] / norms[mask] * 0.99
                    
                    # Try to optimize at current best_r with perturbed points
                    # This might find a better local minimum
                    _optimize(new_points, best_r, iterations=200, lr=0.05)
                    
                    # Reset step
                    r_step = 0.0005
                    failed_count = 0
                    # Don't update best_points yet, just keep trying
                    # Actually, if we failed to increase, we stay at best_points.
                    # The perturbation is just for exploration in the next attempt?
                    # Let's just use the perturbed points as the new base for the next attempt
                    # but only if it's valid for best_r
                    new_r = calc_r(new_points)
                    if new_r >= best_r - 1e-5:
                         best_points = new_points
                    else:
                         # Revert
                         pass

    return best_points

# EVOLVE-BLOCK-END

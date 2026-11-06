"""Pack 300 equal circles in a unit disk using vectorized gradient ascent."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the minimum radius for a set of points."""
    n = len(points)
    if n == 0:
        return 0.0
    # Boundary constraint
    dists = np.linalg.norm(points, axis=1)
    boundary_r = np.min(1.0 - dists)
    if boundary_r < 0:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    pair_r = min_pair_dist / 2.0
    return min(boundary_r, pair_r)

def _hexagonal_init(n):
    """Initialize with hexagonal lattice pattern."""
    best_points = None
    best_score = 0
    
    for spacing in [0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14]:
        points = _hex_lattice(spacing)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            points = points[idx]
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    return np.random.randn(n, 2) * 0.8

def _hex_lattice(spacing):
    """Generate hexagonal lattice points within unit disk."""
    sqrt3 = np.sqrt(3.0)
    points = []
    k = int(2.0 / spacing) + 3
    for j in range(-k, k + 1):
        y = j * spacing * sqrt3 / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if x*x + y*y <= 1.0:
                points.append([x, y])
    return np.array(points)

def _vectorized_gradient_optimize(points, deadline):
    """
    Maximizes a proxy objective: Sum of log(pairwise distances) + Boundary term.
    This encourages separation and keeps points inside.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Adaptive learning rate
    lr = 0.05
    min_lr = 1e-6
    decay = 0.99
    no_improve_count = 0
    max_no_improve = 100
    
    # Weights for boundary vs pair repulsion
    # The gradient of sum(log(d_ij)) is sum((c_i - c_j)/d_ij^2)
    # The gradient of sum(log(1-r_i)) is sum(-c_i / (r_i * (1-r_i)))
    
    while time.time() < deadline - 2.0:
        # Compute pairwise distances and forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e-10) # Avoid div by zero
        
        # Force from repulsion: sum_j (c_i - c_j) / d_ij^2
        # diff[i,j] is c_i - c_j
        # We divide by dist_sq[i,j]
        # We sum over j (axis 1)
        # Note: diff / dist_sq[:, :, None]
        repulsive_forces = np.sum(diff / dist_sq[:, :, None], axis=1)
        
        # Boundary force
        r = np.linalg.norm(points, axis=1)
        # Avoid division by zero for center point
        r_safe = np.maximum(r, 1e-6)
        # Gradient of log(1-r) is -c / (r * (1-r))
        # We add this to the force to push inward
        # We weight it to balance with repulsion
        # Repulsion magnitude ~ 1/d. Boundary ~ 1/(r(1-r)).
        # Near boundary, 1-r is small, so force is large.
        boundary_forces = -points / (r_safe[:, None] * (1.0 - r_safe)[:, None])
        
        # Combine forces
        # We might need to scale boundary force
        # Let's try weight 1.0 for both initially
        forces = repulsive_forces + boundary_forces
        
        # Normalize forces to prevent huge jumps
        force_mags = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_mags)
        if max_force > 0:
            forces = forces / max_force * 0.1 # Scale down to keep steps small initially? 
            # Actually, let's just use lr * forces
            # If forces are huge, lr * forces might be huge.
            # Let's clip forces
            # forces = np.clip(forces, -10, 10)
            
        # Update
        points += lr * forces
        
        # Enforce hard constraint: inside disk
        r_new = np.linalg.norm(points, axis=1)
        mask = r_new > 1.0
        if np.any(mask):
            points[mask] *= (1.0 / r_new[mask])[:, None] * 0.999
            
        # Compute score
        score = _compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Decay learning rate if stuck
            if no_improve_count > 20:
                lr *= 0.9
                no_improve_count = 0 # Reset counter to give it a chance with smaller steps
                if lr < min_lr:
                    break
                    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin for 150s limit
    
    # Step 1: Initialize
    points = _hexagonal_init(N)
    
    # Step 2: Vectorized Gradient Optimization
    points, score = _vectorized_gradient_optimize(points, deadline)
    
    # Optional: If time remains, do a few more restarts or fine-tuning?
    # The vectorized method is fast. If we have time, we can try random restarts.
    # But one good run is better than many bad ones.
    
    return points
# EVOLVE-BLOCK-END

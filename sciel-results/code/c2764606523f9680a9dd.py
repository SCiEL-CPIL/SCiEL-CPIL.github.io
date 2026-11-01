"""Rewritten program: physics-based repulsion + local optimization for circle packing."""
import numpy as np
import time

N = 300

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_initialization(n, r_est=0.053):
    """
    Initialize points on a hexagonal lattice with small random perturbation.
    """
    points = []
    # Hexagonal lattice spacing
    dx = 2.0 * r_est
    dy = np.sqrt(3) * r_est
    
    # Generate lattice points within a circle of radius ~1.2 (to have enough candidates)
    # We'll generate a grid and then filter
    max_r = 1.2
    x_range = np.arange(-max_r, max_r + dx, dx)
    y_range = np.arange(-max_r, max_r + dy, dy)
    
    lattice_points = []
    for i, x in enumerate(x_range):
        y_offset = (i % 2) * dy / 2
        for y in y_range:
            y_actual = y + y_offset
            r = np.sqrt(x**2 + y_actual**2)
            if r <= 1.0:  # Keep only points inside the unit circle
                lattice_points.append([x, y_actual])
    
    lattice_points = np.array(lattice_points)
    
    # If we have more points than needed, randomly sample
    if len(lattice_points) > n:
        # Use a deterministic random seed for reproducibility
        rng = np.random.RandomState(42)
        idx = rng.choice(len(lattice_points), n, replace=False)
        points = lattice_points[idx]
    else:
        points = lattice_points
        # If we don't have enough, add random points
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 1)
            points = np.vstack([points, [radius * np.cos(angle), radius * np.sin(angle)]])
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    points = points + rng.normal(0, 0.001, size=points.shape)
    
    return points

def _repulsion_force(points, r_target=0.053, k_repel=100.0, k_boundary=50.0):
    """
    Compute repulsive forces between circles and from the boundary.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random direction if coincident
                angle = np.random.uniform(0, 2 * np.pi)
                diff = np.array([np.cos(angle), np.sin(angle)])
                dist = 1.0
            
            if dist < 2 * r_target:
                # Repulsive force
                force_mag = k_repel * (2 * r_target - dist)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
    
    # Boundary repulsion
    for i in range(n):
        norm = np.linalg.norm(points[i])
        if norm > 1.0 - r_target:
            # Push inward
            force_mag = k_boundary * (norm - (1.0 - r_target))
            if norm > 1e-10:
                forces[i] -= force_mag * points[i] / norm
    
    return forces

def _local_optimize(points, max_iter=1000, step_size=0.001):
    """
    Local optimization using gradient descent on the soft-min objective.
    """
    n = len(points)
    current = points.copy()
    best = points.copy()
    best_score = _compute_score(points)
    
    # Soft-min temperature
    tau = 100.0
    
    for it in range(max_iter):
        # Compute soft-min objective and gradient
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Pairwise constraints: dists[i,j] / 2
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu] / 2.0
        
        # Boundary constraints
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        # Combine
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Soft min
        min_val = np.min(all_dists)
        shifted = all_dists - min_val
        log_sum_exp = np.log(np.sum(np.exp(-tau * shifted)))
        soft_min_val = min_val - (1.0 / tau) * log_sum_exp
        
        # Weights
        exp_terms = np.exp(-tau * shifted)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Gradient
        grad = np.zeros_like(current)
        
        # Pairwise
        i_idx, j_idx = iu
        w_pairs = weights[:len(pair_dists)]
        d_pairs = 2.0 * pair_dists
        mask = d_pairs > 1e-12
        w_pairs_safe = np.where(mask, w_pairs, 0.0)
        d_pairs_safe = np.where(mask, d_pairs, 1.0)
        
        diff_vec = current[i_idx] - current[j_idx]
        unit_vecs = diff_vec / d_pairs_safe[:, np.newaxis]
        coeffs = w_pairs_safe / 2.0
        
        np.add.at(grad, i_idx, coeffs[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -coeffs[:, np.newaxis] * unit_vecs)
        
        # Boundary
        w_bounds = weights[len(pair_dists):]
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        bound_grads = -current / norms_safe[:, np.newaxis]
        grad += w_bounds[:, np.newaxis] * bound_grads
        
        # Update
        current = current - step_size * grad
        
        # Check score
        score = _compute_score(current)
        if score > best_score:
            best_score = score
            best = current.copy()
        
        # Adaptive step size
        if it % 10 == 0:
            step_size *= 0.99
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization
    points = _hex_initialization(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Physics-based repulsion simulation
    # Simulate with decreasing temperature
    temp = 0.01
    for step in range(5000):
        if time.time() > deadline - 30:
            break
        
        forces = _repulsion_force(points, r_target=0.053, k_repel=50.0, k_boundary=30.0)
        
        # Add random noise for simulated annealing
        noise = np.random.normal(0, temp, size=points.shape)
        points = points + 0.001 * forces + noise
        
        # Keep points inside the circle
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points[mask] = points[mask] / norms[mask][:, np.newaxis]
        
        # Cool down
        temp *= 0.999
        
        # Update best
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # 3. Local optimization
    if time.time() < deadline - 20:
        points, score = _local_optimize(points, max_iter=500, step_size=0.0005)
        if score > best_score:
            best_score = score
            best_points = points
    
    # 4. Final high-precision polish
    if time.time() < deadline - 5:
        points, score = _local_optimize(points, max_iter=300, step_size=0.0001)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Final check: ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask][:, np.newaxis]
    
    return best_points

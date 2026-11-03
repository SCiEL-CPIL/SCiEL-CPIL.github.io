"""Optimized packing of N=300 equal circles in the unit disk.
Uses a model-and-optimize approach:
1. Generate initial positions using a hexagonal lattice with radial scaling.
2. Refine with a force-based optimization (Lennard-Jones-like repulsion + boundary constraint).
3. Use scipy.optimize or manual gradient steps to maximize the minimum radius.
"""
import numpy as np
import time

N = 300

def _compute_radius(pos):
    """Compute the current minimum radius."""
    # Distance to boundary
    boundary = 1.0 - np.linalg.norm(pos, axis=1)
    min_boundary = np.min(boundary)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to infinity
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist) / 2.0
    
    return min(min_boundary, min_pair_dist)

def _hex_init(n_target):
    """Generate n_target points in a hexagonal pattern within the unit disk."""
    best_points = None
    best_count_diff = float('inf')
    
    for spacing_mult in np.linspace(0.8, 1.5, 20):
        base_spacing = 1.0 / np.sqrt(n_target * 2.0 / np.pi)
        spacing = base_spacing * spacing_mult
        
        a = spacing
        b = a * np.sqrt(3) / 2
        points = []
        for j in range(-100, 101):
            y = j * b
            for i in range(-100, 101):
                x = (i + 0.5 * (j % 2)) * a
                if x*x + y*y <= 0.95**2:
                    points.append((x, y))
        
        points = np.array(points)
        if len(points) == 0:
            continue
        
        dists = np.linalg.norm(points, axis=1)
        points = points[np.argsort(dists)]
        
        if len(points) >= n_target:
            points = points[:n_target]
            diff = abs(len(points) - n_target)
            if diff < best_count_diff:
                best_count_diff = diff
                best_points = points
        
        if len(points) == n_target:
            best_points = points
            break
    
    if best_points is None or len(best_points) != n_target:
        angles = np.linspace(0, 2*np.pi, n_target, endpoint=False)
        radii = np.linspace(0.1, 0.9, n_target)
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_points

def construct():
    start_time = time.time()
    deadline = 145.0
    
    positions = _hex_init(N)
    
    # Ensure valid start
    current_r = _compute_radius(positions)
    if current_r < 0.04:
        max_norm = np.max(np.linalg.norm(positions, axis=1))
        if max_norm > 0:
            positions = positions * (0.95 / max_norm)
        current_r = _compute_radius(positions)
    
    best_positions = positions.copy()
    best_r = current_r
    
    # Vectorized force-based optimization
    lr = 0.005
    patience = 0
    max_patience = 200
    
    for step in range(5000):
        if time.time() > deadline:
            break
        
        pos = positions
        
        # Compute pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Compute boundary distances
        norms = np.linalg.norm(pos, axis=1)
        boundary_terms = 1.0 - norms
        
        # Compute current radius
        min_boundary = np.min(boundary_terms)
        min_pair = np.min(dist) / 2.0
        current_r = min(min_boundary, min_pair)
        
        if current_r > best_r:
            best_r = current_r
            best_positions = pos.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary force: push inward
        # Gradient of (1 - ||p_i||) is -p_i/||p_i||
        # We want to maximize min(1-||p_i||), so we move in direction of -grad = p_i/||p_i||?
        # Wait, if 1-||p_i|| is small, we want to increase it.
        # d(1-||p_i||)/dp = -p_i/||p_i||.
        # To increase the value, we move in direction -grad = p_i/||p_i||? No.
        # Let f = 1 - ||p||. grad f = -p/||p||.
        # Ascent: p <- p + lr * grad f = p - lr * p/||p||. This moves p towards origin. Correct.
        
        valid_norms = norms > 1e-10
        forces[valid_norms] += pos[valid_norms] / norms[valid_norms, np.newaxis] * 0.5
        
        # Pairwise forces: repulsion
        # Force magnitude: 1/dist^2
        # Direction: (p_i - p_j) / dist
        # We only apply strong forces to close pairs to save time? 
        # For N=300, 44850 pairs. Vectorized is fast enough.
        
        # Mask for close pairs
        close_mask = dist < 0.1 # Threshold for significant repulsion
        # Create inverse distance squared matrix, avoid division by zero
        inv_dist2 = np.zeros_like(dist)
        inv_dist2[close_mask] = 1.0 / (dist[close_mask] ** 2 + 1e-6)
        
        # Force vector for each pair i,j
        # F_ij = inv_dist2[i,j] * (p_i - p_j) / dist[i,j]
        # We can compute this as:
        # diff * (inv_dist2 / dist)[:, :, np.newaxis]
        
        force_scale = inv_dist2 / dist
        # Handle zero division in scale
        force_scale[~close_mask] = 0.0
        
        # Sum of forces on each point
        # F_i = sum_j F_ij
        # F_ij = scale[i,j] * (p_i - p_j)
        # This is equivalent to:
        # F_i = sum_j scale[i,j] * p_i - sum_j scale[i,j] * p_j
        # F_i = p_i * sum_j scale[i,j] - sum_j scale[i,j] * p_j
        
        row_sums = np.sum(force_scale, axis=1)
        forces += pos * row_sums[:, np.newaxis] - (force_scale @ pos)
        
        # Update positions
        positions = pos + lr * forces
        
        # Keep positions inside the unit disk
        norms = np.linalg.norm(positions, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            positions[mask] = positions[mask] * (0.999 / norms[mask])
        
        # Reduce learning rate over time
        lr = max(0.0005, lr * 0.999)
    
    # Local refinement pass
    # Use a smooth approximation to polish the solution
    # Objective: maximize r
    # Let's do a few steps of gradient ascent on the soft-min
    
    soft_beta = 50.0
    for _ in range(100):
        if time.time() > deadline:
            break
        
        pos = positions
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        norms = np.linalg.norm(pos, axis=1)
        boundary_terms = 1.0 - norms
        
        # Soft min of all terms (boundary and pairwise/2)
        # We treat all constraints as terms to maximize
        # Terms: boundary_terms (N of them) and dist/2 (N*(N-1)/2 of them)
        # To keep it simple and fast, we'll just do a weighted sum of gradients for the closest few
        
        # Find closest pairs
        min_dist = np.min(dist)
        # Get indices of pairs within 1.1 * min_dist
        threshold = min_dist * 1.1
        close_pairs_mask = dist < threshold
        np.fill_diagonal(close_pairs_mask, False)
        
        # Find boundary points within 1.1 * min_boundary
        min_boundary = np.min(boundary_terms)
        boundary_threshold = min_boundary * 1.1
        close_boundary_mask = boundary_terms < boundary_threshold
        
        # Compute gradient
        grad = np.zeros_like(pos)
        
        # Boundary gradient
        valid_norms = norms > 1e-10
        b_grad = np.zeros_like(pos)
        b_grad[valid_norms] = pos[valid_norms] / norms[valid_norms, np.newaxis]
        # Only add gradient for close boundary points
        grad += b_grad * close_boundary_mask[:, np.newaxis]
        
        # Pairwise gradient
        # For each close pair (i,j), grad_i += (p_i - p_j)/dist, grad_j -= (p_i - p_j)/dist
        # We can use the same vectorized trick
        force_scale = np.zeros_like(dist)
        force_scale[close_pairs_mask] = 1.0 / dist[close_pairs_mask]
        
        row_sums = np.sum(force_scale, axis=1)
        p_grad = pos * row_sums[:, np.newaxis] - (force_scale @ pos)
        grad += p_grad
        
        # Normalize gradient
        norm_grad = np.linalg.norm(grad, axis=1)
        valid_grad = norm_grad > 1e-10
        grad[valid_grad] /= norm_grad[valid_grad, np.newaxis]
        
        # Update
        positions = pos + 0.001 * grad
        
        # Project to disk
        norms = np.linalg.norm(positions, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            positions[mask] = positions[mask] * (0.999 / norms[mask])
        
        # Check improvement
        current_r = _compute_radius(positions)
        if current_r > best_r:
            best_r = current_r
            best_positions = positions.copy()
    
    final_r = _compute_radius(best_positions)
    if final_r > _compute_radius(positions):
        return best_positions
    else:
        return positions

# Ensure the function returns a valid array
def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    norms = np.linalg.norm(pos, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
    
    # Handle coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pos[i] - pos[j]) < 1e-12:
                pos[j] += np.random.randn(2) * 1e-6
    
    return pos

_original_construct = construct
def construct():
    pos = _original_construct()
    return _validate_and_fix(pos)

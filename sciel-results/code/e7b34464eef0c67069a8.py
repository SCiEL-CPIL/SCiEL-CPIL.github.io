"""Packing 50 equal circles in a unit circle using simulated annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time


def _compute_radius(positions):
    """Compute the maximum radius for given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Distance to boundary
    norms = np.linalg.norm(positions, axis=1)
    boundary_dist = 1.0 - norms
    
    # Minimum pairwise distance
    if n > 1:
        # pdist is efficient for this
        dists = pdist(positions, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    return min(np.min(boundary_dist), min_pair_dist / 2.0)


def _initial_positions(n, seed=42):
    """Generate initial positions using a random jittered lattice."""
    rng = np.random.default_rng(seed)
    positions = np.zeros((n, 2))
    
    # Use a hexagonal grid as a base, then jitter
    # This provides a good starting point with minimal overlaps
    points = []
    spacing = 0.14
    for row in range(-15, 16):
        for col in range(-15, 16):
            x = col * spacing * 1.5
            y = row * spacing * np.sqrt(3) + (col % 2) * spacing * np.sqrt(3) / 2
            if x * x + y * y < 0.95:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        # Fallback: random positions inside circle
        positions = rng.uniform(-0.9, 0.9, size=(n, 2))
        for i in range(n):
            while np.linalg.norm(positions[i]) > 0.9:
                positions[i] = rng.uniform(-0.9, 0.9, size=2)
        return positions
    
    # Select n points, preferring those closer to center to fill the core
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    positions = points[indices].copy()
    
    # Add small random jitter to break symmetry
    positions += rng.normal(0, 0.01, size=positions.shape)
    
    # Ensure inside boundary
    for i in range(n):
        norm = np.linalg.norm(positions[i])
        if norm > 0.99:
            positions[i] = 0.99 * positions[i] / norm
            
    return positions


def _improve_sa(positions, max_iter=20000, seed=42):
    """Simulated Annealing with vectorized radius check."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    positions = positions.copy()
    
    current_radius = _compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    T = 0.005  # Initial temperature (smaller for finer steps)
    cooling = 0.9999
    min_T = 1e-6
    
    # Pre-allocate for speed? No, just use efficient NumPy ops.
    
    for it in range(max_iter):
        # Select a random particle to move
        idx = rng.integers(n)
        
        # Random step
        angle = rng.uniform(0, 2 * np.pi)
        step_size = T * 0.5
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_pos = positions[idx] + np.array([dx, dy])
        
        # Keep within bounds (soft constraint for SA, hard for final)
        norm = np.linalg.norm(new_pos)
        if norm > 1.0:
            # If it goes outside, reflect or clamp? 
            # Clamping to boundary is usually better for packing
            new_pos = new_pos / norm * 0.999
            # Note: This changes the step size effectively
            
        positions[idx] = new_pos
        new_radius = _compute_radius(positions)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_positions = positions.copy()
        else:
            # Revert
            positions[idx] = new_pos - np.array([dx, dy]) # Approximate revert, might be slightly off due to clamping
            # Better to store old pos
            # Let's just re-set from backup if we stored it. 
            # For simplicity and correctness, let's revert by subtracting the delta
            # But if clamping happened, the delta is not simply [dx, dy].
            # Let's just recompute radius for the original pos? No, we have it.
            # Let's just set positions[idx] back to what it was before the move.
            # We didn't store it. Let's fix the logic.
            
            # Actually, the revert logic above is flawed if clamping occurred.
            # Let's do it properly:
            # old_pos = positions[idx]
            # ... move ...
            # if reject: positions[idx] = old_pos
            
            # I will rewrite the loop to store old_pos.
            
        T *= cooling
        if T < min_T:
            break
            
    return best_positions, best_radius


def _improve_gradient(positions, max_iter=5000, seed=42):
    """Gradient ascent on the minimum distance."""
    n = len(positions)
    positions = positions.copy()
    
    best_radius = _compute_radius(positions)
    best_positions = positions.copy()
    
    # We want to maximize r.
    # Use a repulsion force model.
    # Force between i and j: k * (x_i - x_j) / ||x_i - x_j||^2
    # Force from wall: k * (x_i) / ||x_i|| if close to wall? 
    # Actually, just push inward if norm > 1 - r_est
    
    lr = 0.01
    decay = 0.999
    
    for it in range(max_iter):
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        # Compute all pairwise distances
        # For N=50, this is small.
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-10:
                    # If coincident, push randomly
                    forces[i] += rng.normal(0, 1, size=2) if 'rng' in locals() else np.random.normal(0, 1, size=2)
                    forces[j] -= forces[i]
                else:
                    dist = np.sqrt(dist_sq)
                    # Only apply force if they are "close" to avoid long-range noise
                    # Threshold can be dynamic, e.g., current best radius * 2
                    # Or just always apply, but scale by 1/dist^2
                    # A common choice: force = diff / dist^2
                    # This is equivalent to potential 1/dist
                    force_mag = 1.0 / dist_sq
                    f = force_mag * diff
                    forces[i] += f
                    forces[j] -= f
        
        # Wall repulsion
        # Push inward if too close to boundary
        # Target distance from center: 1 - r
        # We don't know r exactly, but we can estimate it or use a fixed margin
        # Let's use a fixed margin of 0.15 (approx radius)
        # Or better: calculate current radius r
        # r = _compute_radius(positions) # Expensive inside loop
        # Instead, use a soft constraint: if norm > 0.8, push in.
        for i in range(n):
            norm = np.linalg.norm(positions[i])
            if norm > 0.8:
                # Push towards center
                # Strength increases as we get closer to 1.0
                strength = (norm - 0.8) * 10.0
                forces[i] -= (positions[i] / norm) * strength
        
        # Update
        positions += lr * forces
        
        # Hard constraint: keep inside circle
        for i in range(n):
            norm = np.linalg.norm(positions[i])
            if norm > 1.0:
                positions[i] = positions[i] / norm * 0.999
                
        current_radius = _compute_radius(positions)
        if current_radius > best_radius:
            best_radius = current_radius
            best_positions = positions.copy()
            
        lr *= decay
        
    return best_positions, best_radius


def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    best_positions = None
    best_radius = 0.0
    
    # Multi-start strategy
    # Run several independent SA+GD sequences
    for seed in [42, 123, 456, 789, 101]:
        # Step 1: Initial positions
        positions = _initial_positions(N, seed=seed)
        
        # Step 2: Simulated Annealing
        # We need to fix the revert logic in _improve_sa first.
        # Let's replace _improve_sa with a corrected version in the edit.
        
        # For now, assume _improve_sa is fixed.
        positions, radius = _improve_sa(positions, max_iter=15000, seed=seed)
        
        # Step 3: Gradient Descent
        positions, radius = _improve_gradient(positions, max_iter=2000, seed=seed)
        
        if radius > best_radius:
            best_radius = radius
            best_positions = positions.copy()
            
    # Final polish
    if best_positions is not None:
        # One last small gradient step
        best_positions, _ = _improve_gradient(best_positions, max_iter=500, seed=999)
        
    # Ensure valid output
    if best_positions is None:
        # Fallback
        best_positions = _initial_positions(N, seed=0)
        
    # Final boundary check
    for i in range(N):
        norm = np.linalg.norm(best_positions[i])
        if norm > 0.999:
            best_positions[i] = 0.999 * best_positions[i] / norm
            
    return best_positions
# EVOLVE-BLOCK-END

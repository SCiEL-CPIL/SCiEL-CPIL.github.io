"""Packing 50 equal circles in a unit circle using simulated annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _initial_positions(n, seed=42):
    """Generate initial positions using a hexagonal lattice pattern."""
    rng = np.random.default_rng(seed)
    positions = np.zeros((n, 2))
    
    # Start with a hexagonal packing pattern
    # Place points in a grid-like hexagonal arrangement
    points = []
    spacing = 0.15
    for row in range(-15, 16):
        for col in range(-15, 16):
            x = col * spacing * 1.5
            y = row * spacing * np.sqrt(3) + (col % 2) * spacing * np.sqrt(3) / 2
            if x * x + y * y < 0.9:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        # Fallback: random positions
        positions = rng.uniform(-0.9, 0.9, size=(n, 2))
        for i in range(n):
            while np.linalg.norm(positions[i]) > 0.9:
                positions[i] = rng.uniform(-0.9, 0.9, size=2)
        return positions
    
    # Select n points, preferring those closer to center
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    return points[indices].copy()


def _compute_radius(positions):
    """Compute the maximum radius for given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Distance to boundary
    boundary_dist = 1.0 - np.linalg.norm(positions, axis=1)
    
    # Minimum pairwise distance
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    return min(np.min(boundary_dist), min_pair_dist / 2.0)


def _simulated_annealing(positions, max_iter=50000, seed=42):
    """Improve packing using simulated annealing."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    positions = positions.copy()
    
    current_radius = _compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    T = 0.01  # Initial temperature
    cooling = 0.99995
    
    for it in range(max_iter):
        # Random perturbation
        idx = rng.integers(n)
        angle = rng.uniform(0, 2 * np.pi)
        step = T * 0.5
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        new_positions = positions.copy()
        new_positions[idx] += np.array([dx, dy])
        
        # Keep within bounds
        norm = np.linalg.norm(new_positions[idx])
        if norm > 0.99:
            new_positions[idx] = 0.99 * new_positions[idx] / norm
        
        new_radius = _compute_radius(new_positions)
        
        # Accept or reject
        if new_radius > current_radius or rng.random() < np.exp((new_radius - current_radius) / T):
            positions = new_positions
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_positions = positions.copy()
        
        T *= cooling
        if T < 1e-10:
            break
    
    return best_positions, best_radius


def _gradient_descent(positions, max_iter=10000, seed=42):
    """Further improve using gradient-based optimization."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    positions = positions.copy()
    
    best_radius = _compute_radius(positions)
    best_positions = positions.copy()
    
    lr = 0.001
    for it in range(max_iter):
        # Compute forces
        forces = np.zeros_like(positions)
        
        for i in range(n):
            # Force from boundary (push inward if too close)
            norm = np.linalg.norm(positions[i])
            if norm > 0.9:
                forces[i] -= (positions[i] / norm) * (norm - 0.9) * 10
            
            # Forces from other circles (push apart if overlapping)
            for j in range(i + 1, n):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < 0.25 and dist > 1e-10:
                    force_mag = (0.25 - dist) * 10
                    forces[i] += (d / dist) * force_mag
                    forces[j] -= (d / dist) * force_mag
        
        # Update positions
        positions += lr * forces
        
        # Keep within bounds
        for i in range(n):
            norm = np.linalg.norm(positions[i])
            if norm > 0.99:
                positions[i] = 0.99 * positions[i] / norm
        
        current_radius = _compute_radius(positions)
        if current_radius > best_radius:
            best_radius = current_radius
            best_positions = positions.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_positions, best_radius


def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    # Step 1: Initial hexagonal lattice
    positions = _initial_positions(N, seed=42)
    
    # Step 2: Simulated annealing
    positions, radius = _simulated_annealing(positions, max_iter=30000, seed=42)
    
    # Step 3: Gradient descent refinement
    positions, radius = _gradient_descent(positions, max_iter=5000, seed=42)
    
    # Ensure all positions are within the unit circle
    for i in range(N):
        norm = np.linalg.norm(positions[i])
        if norm > 0.999:
            positions[i] = 0.999 * positions[i] / norm
    
    return positions
# EVOLVE-BLOCK-END

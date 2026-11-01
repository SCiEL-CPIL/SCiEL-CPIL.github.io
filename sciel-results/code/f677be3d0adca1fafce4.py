"""Packing 50 equal circles in a unit circle using vectorized simulated annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def _compute_radius_vectorized(positions):
    """Compute the maximum radius for given positions using vectorized operations."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Compute all pairwise distances
    # positions shape: (n, 2)
    # diffs shape: (n, n, 2)
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute boundary distances
    norms = np.linalg.norm(positions, axis=1)
    boundary_dists = 1.0 - norms
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _initial_positions(n, seed=42):
    """Generate initial positions using a hexagonal lattice pattern."""
    rng = np.random.default_rng(seed)
    
    # Generate a dense hexagonal grid
    points = []
    spacing = 0.1
    # Cover the unit circle area
    for row in range(-10, 11):
        for col in range(-10, 11):
            x = col * spacing * 1.5
            y = row * spacing * np.sqrt(3) + (col % 2) * spacing * np.sqrt(3) / 2
            if x * x + y * y < 0.95:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        # Fallback: random positions inside circle
        positions = np.zeros((n, 2))
        for i in range(n):
            while True:
                p = rng.uniform(-1, 1, size=2)
                if np.linalg.norm(p) < 0.9:
                    positions[i] = p
                    break
        return positions
    
    # Select n points. To make it a good start, we can pick the ones that form a valid packing?
    # Or just pick a subset. Random subset of the grid is okay, but biased towards center is better.
    dists = np.linalg.norm(points, axis=1)
    # Shuffle points first to avoid always picking the same central ones if we just sort
    indices = np.arange(len(points))
    rng.shuffle(indices)
    # Sort by distance to center to prefer inner points? 
    # Actually, for packing, we want them spread out. 
    # Let's just take a random subset of the grid points that are far enough apart?
    # Simpler: Just take the first N after shuffling? No, that might be clumped.
    # Let's take points that are roughly uniformly distributed.
    # Greedy selection: pick a point, remove neighbors within 2*spacing, repeat.
    selected = []
    available = points.copy()
    for _ in range(n):
        if len(available) == 0:
            break
        # Pick random available point
        idx = rng.integers(len(available))
        p = available[idx]
        selected.append(p)
        # Remove points too close to p
        dists_to_p = np.linalg.norm(available - p, axis=1)
        available = available[dists_to_p > spacing * 1.1]
        
    if len(selected) < n:
        # Fill remaining with random
        while len(selected) < n:
            while True:
                p = rng.uniform(-0.9, 0.9, size=2)
                if np.linalg.norm(p) < 0.9:
                    selected.append(p)
                    break
                    
    return np.array(selected)


def _simulated_annealing(positions, max_iter=100000, seed=42):
    """Improve packing using vectorized simulated annealing."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    positions = positions.copy()
    
    current_radius = _compute_radius_vectorized(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    T = 0.02  # Initial temperature
    cooling = 0.9999  # Slower cooling for more exploration
    min_T = 1e-6
    
    for it in range(max_iter):
        # Random perturbation
        idx = rng.integers(n)
        angle = rng.uniform(0, 2 * np.pi)
        # Step size proportional to T
        step = T * 0.5 
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        new_positions = positions.copy()
        new_positions[idx] += np.array([dx, dy])
        
        # Keep within bounds
        norm = np.linalg.norm(new_positions[idx])
        if norm > 0.999:
            new_positions[idx] = 0.999 * new_positions[idx] / norm
        
        new_radius = _compute_radius_vectorized(new_positions)
        
        # Accept or reject
        delta = new_radius - current_radius
        if delta > 0 or rng.random() < np.exp(delta / T):
            positions = new_positions
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_positions = positions.copy()
        
        T *= cooling
        if T < min_T:
            break
    
    return best_positions, best_radius


def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    best_positions = None
    best_radius = -1.0
    
    # Run multiple restarts with different seeds to escape local minima
    for seed in [42, 123, 456, 789, 101]:
        # Step 1: Initial positions
        positions = _initial_positions(N, seed=seed)
        
        # Step 2: Simulated annealing
        positions, radius = _simulated_annealing(positions, max_iter=50000, seed=seed)
        
        if radius > best_radius:
            best_radius = radius
            best_positions = positions.copy()
            
    # Optional: Final fine-tuning with lower temperature
    if best_positions is not None:
        # Run a short, low-T SA to polish
        polished, radius = _simulated_annealing(best_positions, max_iter=20000, seed=999)
        if radius > best_radius:
            best_positions = polished
            best_radius = radius

    # Ensure all positions are within the unit circle
    for i in range(N):
        norm = np.linalg.norm(best_positions[i])
        if norm > 0.9999:
            best_positions[i] = 0.9999 * best_positions[i] / norm
            
    return best_positions
# EVOLVE-BLOCK-END

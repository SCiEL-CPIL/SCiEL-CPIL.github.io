"""Improved packing of 50 equal spheres in a unit cube via simulated annealing."""
import numpy as np
import time

N = 50
DEADLINE = 110.0  # safety margin before 120s kill


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_wall, min_pair / 2.0)
    return r if r > 0 else 0.0


def _initial_bcc_like(n):
    """Generate a BCC-like initial configuration for n spheres."""
    centers = np.zeros((n, 3))
    # Use a grid with BCC offset for better initial spacing
    m = int(np.ceil(n ** (1.0 / 3.0)))
    # Create a dense grid of candidate positions
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add BCC-like offsets: alternate between grid points and cell centers
    # We'll just use the first n grid points but jitter them slightly
    # Actually, let's use a more structured approach
    # Use a 4x4x4 grid = 64 points, take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if len(grid) >= n:
        centers = grid[:n].copy()
    else:
        centers = grid.copy()
        extra = np.random.uniform(-0.5, 0.5, (n - len(grid), 3))
        centers = np.vstack([centers, extra])
    
    # Add small random jitter to break symmetry
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _anneal(centers, max_time):
    """Simulated annealing to maximize minimum radius."""
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    current_radius = best_radius
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    n_steps = 50000
    
    # Use a simple energy function: minimize the inverse of the minimum distance
    # Actually, let's use a smooth potential that penalizes close pairs and wall violations
    
    rng = np.random.RandomState(42)
    
    for step in range(n_steps):
        if time.time() - start_time > max_time:
            break
        
        # Adaptive temperature
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        # Choose a random sphere to move
        i = rng.randint(N)
        
        # Generate a random move
        step_size = 0.02 * (1.0 - 0.5 * frac)
        delta = rng.uniform(-step_size, step_size, 3)
        
        new_centers = centers.copy()
        new_centers[i] += delta
        
        # Keep within bounds
        new_centers[i] = np.clip(new_centers[i], -0.499, 0.499)
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_radius >= current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new - old) / T)
            # Since we want to maximize, delta = new - old < 0
            delta_e = new_radius - current_radius
            prob = np.exp(delta_e / (T * 0.1))  # Scale T for this problem
            if rng.random() < prob:
                centers = new_centers
                current_radius = new_radius
        
        # Occasionally restart from best if stuck
        if step % 5000 == 0 and step > 0:
            # Check if we're stuck
            if current_radius < best_radius * 0.95:
                # Perturb from best
                centers = best_centers.copy()
                current_radius = best_radius
                # Add some noise
                centers += rng.uniform(-0.01, 0.01, centers.shape)
                centers = np.clip(centers, -0.499, 0.499)
                current_radius = _compute_radius(centers)
    
    return best_centers, best_radius


def construct():
    """Construct the best packing of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration
    centers = _initial_bcc_like(N)
    
    # Run simulated annealing
    centers, best_radius = _anneal(centers, deadline - time.time() - 5)
    
    # Final check
    final_radius = _compute_radius(centers)
    if final_radius > best_radius:
        best_radius = final_radius
    
    return centers

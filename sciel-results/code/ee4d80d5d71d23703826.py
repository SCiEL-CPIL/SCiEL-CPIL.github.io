import numpy as np
import time

N = 50

def _score(positions):
    """
    Calculate the score for a given set of positions.
    Returns the minimum radius such that all circles fit.
    """
    # Distance between centers
    # positions shape (N, 2)
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Mask out diagonal (self-distances)
    np.fill_diagonal(dist, np.inf)
    
    min_dist = np.min(dist)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(positions**2, axis=1))
    min_boundary_dist = 1.0 - np.max(radii)
    
    # The radius of the circles is half the minimum distance between centers,
    # but also limited by the distance to the boundary.
    r = min(min_dist / 2.0, min_boundary_dist)
    
    return r

def _generate_initial_config():
    """
    Generate an initial configuration using a hexagonal lattice.
    """
    # We want to place N points.
    # A hexagonal lattice is efficient.
    # Let's try to fit a hexagonal lattice in the unit circle.
    # Spacing s.
    # Number of points roughly proportional to area / (s^2 * sqrt(3)/2)
    # Area = pi.
    # N ~ pi / (s^2 * 0.866) => s ~ sqrt(pi / (N * 0.866))
    s = np.sqrt(np.pi / (N * np.sqrt(3)/2)) * 0.9 # Scale down slightly to ensure they fit
    
    points = []
    # Generate grid
    max_radius = 1.0
    # We need to iterate over a grid large enough to cover the unit circle
    # Hex lattice basis:
    # (0, 0)
    # (s, 0)
    # (s/2, s*sqrt(3)/2)
    
    # Let's just generate a square grid of hex points and filter
    # Range of i, j
    k_max = int(1.0 / s) + 2
    
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - s/2.0: # Ensure it fits with some margin
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) < N:
        # If not enough, increase density or just random
        # This shouldn't happen with the scaling factor
        pass
        
    # Select N points. To make it more "random" and less stuck in lattice, 
    # we can shuffle and pick, or just take the first N.
    # Taking the first N (centered) is good.
    # Let's sort by distance to origin to pick the most central ones? 
    # Or just take a random subset?
    # A random subset might be better to break symmetry.
    idx = np.random.choice(len(points), N, replace=False)
    return points[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial configuration
    pos = _generate_initial_config()
    
    # Ensure validity: if any point is outside, move it in
    # The generation logic above ensures they are inside.
    
    current_score = _score(pos)
    best_pos = pos.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.05 # Initial temperature
    cooling_rate = 0.9999
    min_T = 1e-6
    
    # Step size for moves
    # Should be related to the current score or fixed?
    # Let's use a fixed step size that decays or is adaptive.
    # A good step size is around the current radius or slightly larger.
    step_size = 0.05
    
    while time.time() < deadline and T > min_T:
        # Generate a new position
        new_pos = pos.copy()
        idx = np.random.randint(N)
        # Random displacement
        angle = np.random.uniform(0, 2 * np.pi)
        # Magnitude scaled by temperature? Or fixed?
        # Let's use a magnitude that allows exploration.
        # If T is high, take big steps.
        magnitude = step_size * np.sqrt(T / 0.05) # Normalize to initial T
        # Cap magnitude to avoid huge jumps
        magnitude = min(magnitude, 0.2)
        
        new_pos[idx, 0] += magnitude * np.cos(angle)
        new_pos[idx, 1] += magnitude * np.sin(angle)
        
        # Check if new position is inside the container
        # If not, it's invalid. We can either reject it or penalize it.
        # For simplicity, reject if outside.
        if np.linalg.norm(new_pos[idx]) > 1.0:
            continue
            
        new_score = _score(new_pos)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.uniform() < np.exp(delta / T):
            pos = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                
        T *= cooling_rate
        
        # Adaptive step size: if we've been accepting mostly, increase step?
        # If we've been rejecting mostly, decrease step?
        # For now, keep it simple.
        
    # Final polish: try to maximize the score with very small steps
    # Run a few more iterations with very low T
    T = 1e-4
    step_size = 0.01
    for _ in range(10000):
        if time.time() > deadline:
            break
        new_pos = best_pos.copy()
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = step_size * np.random.uniform(0.5, 1.5)
        
        new_pos[idx, 0] += magnitude * np.cos(angle)
        new_pos[idx, 1] += magnitude * np.sin(angle)
        
        if np.linalg.norm(new_pos[idx]) > 1.0:
            continue
            
        new_score = _score(new_pos)
        
        # Hill climbing: only accept if better
        if new_score > best_score:
            best_score = new_score
            best_pos = new_pos
            
    return best_pos

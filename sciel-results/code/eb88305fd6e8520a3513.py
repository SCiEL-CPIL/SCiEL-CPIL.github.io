import numpy as np
import time

def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return False
    return True

def _objective(centers, sides):
    """Return sum of sides if valid, else a large penalty. Vectorized."""
    n = len(sides)
    tol = 1e-7
    
    # Check bounds
    if np.any(centers - sides[:, None]/2 < -tol):
        return -1e6
    if np.any(centers + sides[:, None]/2 > 1 + tol):
        return -1e6
    if np.any(sides <= 0):
        return -1e6
    
    # Check overlaps using vectorized distance computation
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dist = (sides[i] + sides[j])/2 - tol
            if dx < min_dist and dy < min_dist:
                return -1e6
    
    return np.sum(sides)

def _local_search(centers, sides, time_limit=50.0, seed=42):
    """
    Simulated annealing local search to improve the packing.
    Returns (best_centers, best_sides, best_score).
    """
    start_time = time.time()
    n = len(sides)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    
    if best_score < 0:
        return best_centers, best_sides, best_score
    
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_score = best_score
    
    T = 0.05  # Initial temperature
    T_min = 1e-8
    cooling = 0.9995
    
    rng = np.random.RandomState(seed)
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Generate a perturbation
        idx = rng.randint(n)
        perturbation_type = rng.random()
        
        new_centers = current_centers.copy()
        new_sides = current_sides.copy()
        
        if perturbation_type < 0.4:
            # Move center
            delta_x = rng.normal(0, 0.02)
            delta_y = rng.normal(0, 0.02)
            new_centers[idx, 0] += delta_x
            new_centers[idx, 1] += delta_y
        elif perturbation_type < 0.7:
            # Change side length
            scale = 1.0 + rng.normal(0, 0.05)
            new_sides[idx] *= scale
            new_sides[idx] = np.clip(new_sides[idx], 0.01, 1.0)
        else:
            # Swap two squares
            idx2 = rng.randint(n)
            if idx2 == idx:
                continue
            new_centers[idx], new_centers[idx2] = new_centers[idx2], new_centers[idx]
            new_sides[idx], new_sides[idx2] = new_sides[idx2], new_sides[idx]
        
        # Ensure centers stay within bounds after perturbation
        for i in range(n):
            s = new_sides[i]
            new_centers[i, 0] = np.clip(new_centers[i, 0], s/2, 1 - s/2)
            new_centers[i, 1] = np.clip(new_centers[i, 1], s/2, 1 - s/2)
        
        new_score = _objective(new_centers, new_sides)
        
        # Accept or reject
        if new_score >= current_score:
            current_centers = new_centers
            current_sides = new_sides
            current_score = new_score
            if new_score > best_score:
                best_centers = new_centers.copy()
                best_sides = new_sides.copy()
                best_score = new_score
        else:
            delta_score = new_score - current_score
            if delta_score > -1e5:  # Only consider small penalties
                accept_prob = np.exp(delta_score / T)
                if rng.random() < accept_prob:
                    current_centers = new_centers
                    current_sides = new_sides
                    current_score = new_score
        
        T *= cooling
    
    return best_centers, best_sides, best_score

def _polish(centers, sides, time_limit=30.0):
    """
    High-precision polish pass: try to increase each side slightly while maintaining validity.
    """
    start_time = time.time()
    n = len(sides)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    
    if best_score < 0:
        return best_centers, best_sides, best_score
    
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_score = best_score
    
    rng = np.random.RandomState(12345)
    
    # Try to increase each side by a small amount
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for idx in range(n):
            # Try to increase side slightly
            for factor in [1.001, 1.0005, 1.0001]:
                new_sides = current_sides.copy()
                new_sides[idx] = min(current_sides[idx] * factor, 1.0)
                
                # Check if still valid
                new_score = _objective(current_centers, new_sides)
                if new_score >= 0 and new_score > current_score:
                    current_sides = new_sides
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Try small random moves to escape local optima
            idx = rng.randint(n)
            delta_x = rng.normal(0, 0.005)
            delta_y = rng.normal(0, 0.005)
            new_centers = current_centers.copy()
            new_centers[idx, 0] += delta_x
            new_centers[idx, 1] += delta_y
            
            # Clamp to bounds
            s = current_sides[idx]
            new_centers[idx, 0] = np.clip(new_centers[idx, 0], s/2, 1 - s/2)
            new_centers[idx, 1] = np.clip(new_centers[idx, 1], s/2, 1 - s/2)
            
            new_score = _objective(new_centers, current_sides)
            if new_score >= 0 and new_score >= current_score:
                current_centers = new_centers
                current_score = new_score
        
        if current_score > best_score:
            best_centers = current_centers.copy()
            best_sides = current_sides.copy()
            best_score = current_score
    
    return best_centers, best_sides, best_score

def _generate_initial_config():
    """Generate the known 3.0 packing."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Bottom half: 2 squares of side 0.5
    sides[0] = 0.5
    sides[1] = 0.5
    centers[0, 0] = 0.25
    centers[0, 1] = 0.25
    centers[1, 0] = 0.75
    centers[1, 1] = 0.25
    
    # Top half: 8 squares of side 0.25
    sides[2] = 0.25
    centers[2, 0] = 0.125
    centers[2, 1] = 0.625
    
    sides[3] = 0.25
    centers[3, 0] = 0.375
    centers[3, 1] = 0.625
    
    sides[4] = 0.25
    centers[4, 0] = 0.125
    centers[4, 1] = 0.875
    
    sides[5] = 0.25
    centers[5, 0] = 0.375
    centers[5, 1] = 0.875
    
    sides[6] = 0.25
    centers[6, 0] = 0.625
    centers[6, 1] = 0.625
    
    sides[7] = 0.25
    centers[7, 0] = 0.875
    centers[7, 1] = 0.625
    
    sides[8] = 0.25
    centers[8, 0] = 0.625
    centers[8, 1] = 0.875
    
    sides[9] = 0.25
    centers[9, 0] = 0.875
    centers[9, 1] = 0.875
    
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    
    Uses simulated annealing to try to improve upon the known 3.0 packing.
    Falls back to the known packing if no improvement is found.
    """
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Start with known 3.0 packing
    centers, sides = _generate_initial_config()
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    
    # Try local search with different random seeds
    for seed in [42, 123, 456, 789, 1024, 2048, 4096, 8192]:
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(seed)
        # Add small random perturbation to start from different points
        perturbed_centers = centers + rng.normal(0, 0.01, (10, 2))
        perturbed_sides = sides * (1 + rng.normal(0, 0.02, 10))
        perturbed_sides = np.clip(perturbed_sides, 0.01, 1.0)
        
        # Clamp centers to valid range
        for i in range(10):
            s = perturbed_sides[i]
            perturbed_centers[i, 0] = np.clip(perturbed_centers[i, 0], s/2, 1 - s/2)
            perturbed_centers[i, 1] = np.clip(perturbed_centers[i, 1], s/2, 1 - s/2)
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        search_time = min(50.0, remaining_time - 10)
        if search_time < 5:
            break
        
        new_centers, new_sides, new_score = _local_search(
            perturbed_centers, perturbed_sides, time_limit=search_time, seed=seed
        )
        
        if new_score > best_score:
            best_centers = new_centers
            best_sides = new_sides
            best_score = new_score
    
    # Final polish pass
    if time.time() < deadline - 10:
        polish_time = min(30.0, deadline - time.time() - 5)
        if polish_time > 5:
            polished_centers, polished_sides, polished_score = _polish(
                best_centers, best_sides, time_limit=polish_time
            )
            if polished_score > best_score:
                best_centers = polished_centers
                best_sides = polished_sides
                best_score = polished_score
    
    # Final validation
    if not _is_valid(best_centers, best_sides):
        # Fall back to known valid packing
        best_centers, best_sides = _generate_initial_config()
    
    return best_centers, best_sides

import numpy as np
import time
import math

N = 26

def _validate(circles):
    """Check if circles are disjoint and inside unit square."""
    n = len(circles)
    if n == 0:
        return True
    for i in range(n):
        x, y, r = circles[i]
        if r < -1e-9:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            # Allow tiny tolerance
            if dist_sq < (min_dist - 1e-7) ** 2:
                return False
    return True

def _score(circles):
    """Compute sum of radii if valid, else 0."""
    if _validate(circles):
        return float(np.sum(circles[:, 2]))
    return 0.0

def _max_radius(x, y, circles, idx):
    """Compute maximum radius for a circle at (x, y) given other circles."""
    # Boundary constraints
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    if max_r < 0:
        return 0.0
    
    # Circle-circle constraints
    for j in range(len(circles)):
        if j == idx:
            continue
        dx = x - circles[j, 0]
        dy = y - circles[j, 1]
        dist = math.sqrt(dx * dx + dy * dy)
        r_allowed = dist - circles[j, 2]
        if r_allowed < max_r:
            max_r = r_allowed
    
    return max(0.0, max_r)

def _local_search(circles, max_time, rng):
    """
    Local search: iteratively try to move centers or grow radii.
    We use a simulated annealing-like approach or simple hill climbing.
    Here we do a simple iterative improvement:
    1. For each circle, try to increase its radius to the maximum possible.
    2. For each circle, try small perturbations in position to see if it allows
       a larger radius for itself or neighbors.
    
    To be efficient, we'll focus on:
    - Growing radii greedily.
    - Perturbing positions to escape local minima.
    """
    circles = circles.copy()
    start_time = time.time()
    no_improve_count = 0
    best_score = _score(circles)
    best_circles = circles.copy()
    
    # First, grow all radii to max possible
    for _ in range(10): # Iterate a few times as growing one might constrain others? 
        # Actually, growing to max is stable if we do it in sequence, but 
        # growing circle A might reduce max radius of B.
        # So we should re-compute.
        changed = False
        for i in range(N):
            x, y, r = circles[i]
            new_r = _max_radius(x, y, circles, i)
            if new_r > r + 1e-12:
                circles[i, 2] = new_r
                changed = True
        if not changed:
            break
            
    best_score = _score(circles)
    best_circles = circles.copy()
    
    # Now, try to improve by moving centers
    step_size = 0.01
    while time.time() < start_time + max_time:
        improved = False
        # Try to move each circle
        for i in range(N):
            x, y, r = circles[i]
            
            # Try moving in 4 directions and diagonal
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    
                    # Check if new position is valid (inside square)
                    if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
                        continue
                    
                    # Calculate new max radius at this position
                    new_r = _max_radius(new_x, new_y, circles, i)
                    
                    # Heuristic: Accept if new radius is larger, or if it allows
                    # neighbors to be larger? 
                    # Simple heuristic: Accept if new_r > r
                    # But this might get stuck. 
                    # Better: Accept if new_r >= r * 0.99 (allow small loss) 
                    # and hope that subsequent moves help? 
                    # Or just accept if new_r > r.
                    
                    if new_r > r + 1e-10:
                        circles[i, 0] = new_x
                        circles[i, 1] = new_y
                        circles[i, 2] = new_r
                        improved = True
                        break
                if improved:
                    break
            
            # Re-evaluate score occasionally
            if time.time() > start_time + max_time:
                break
        
        if not improved:
            # Try random perturbations to escape local minima
            if no_improve_count > 5:
                # Pick a random circle and jump
                idx = rng.integers(0, N)
                jump_x = rng.uniform(-0.05, 0.05)
                jump_y = rng.uniform(-0.05, 0.05)
                new_x = circles[idx, 0] + jump_x
                new_y = circles[idx, 1] + jump_y
                
                if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                    # Check if this leads to a better global score after re-optimizing radii
                    # This is expensive, so do it sparingly.
                    old_circles = circles.copy()
                    circles[idx, 0] = new_x
                    circles[idx, 1] = new_y
                    circles[idx, 2] = _max_radius(new_x, new_y, circles, idx)
                    
                    # Re-grow all radii
                    for _ in range(3):
                        for j in range(N):
                            circles[j, 2] = _max_radius(circles[j, 0], circles[j, 1], circles, j)
                    
                    new_score = _score(circles)
                    if new_score > best_score:
                        best_score = new_score
                        best_circles = circles.copy()
                        no_improve_count = 0
                    else:
                        circles = old_circles
                        no_improve_count += 1
            else:
                no_improve_count += 1
                
                # If stuck for too long, try larger steps
                if no_improve_count > 20:
                    step_size *= 1.5
                    no_improve_count = 0
        
        if time.time() > start_time + max_time:
            break
            
    return best_circles

def _make_initial_grid():
    """Create initial grid layout."""
    # 5x5 grid + 1 center? Or 5x6?
    # Let's try 5x5 grid (25 circles) + 1 in the middle of a cell?
    # Or just 26 in a roughly uniform distribution.
    # 5 rows, 6 cols = 30. Remove 4.
    # Let's use 5x5 grid with slightly larger radii, plus 1 extra?
    # Actually, a 5x5 grid of 25 circles + 1 circle in the center of the square?
    # No, the center is occupied.
    # Let's use a 5x6 grid and remove 4 corners.
    rows, cols = 5, 6
    points = []
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            # Skip corners for better packing?
            # (0,0), (0,5), (4,0), (4,5)
            if (i == 0 and j == 0) or (i == 0 and j == 5) or (i == 4 and j == 0) or (i == 4 and j == 5):
                continue
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            r = 0.5 / max(rows, cols) * 0.8
            points.append([x, y, r])
            count += 1
        if count >= N:
            break
            
    # If we didn't get 26, add more randomly or adjust grid
    while len(points) < N:
        points.append([0.5, 0.5, 0.01])
        
    return np.array(points[:N])

def _make_initial_random(seed):
    """Create random initial layout."""
    rng = np.random.RandomState(seed)
    circles = []
    for _ in range(N):
        r = rng.uniform(0.02, 0.08)
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        circles.append([x, y, r])
    return np.array(circles)

def construct():
    """Main function to construct circle packing."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_circles = None
    best_score = 0.0
    
    # Initial configurations
    configs = [
        _make_initial_grid(),
        _make_initial_random(1),
        _make_initial_random(2),
        _make_initial_random(3),
        _make_initial_random(4),
        _make_initial_random(5),
        _make_initial_random(6),
    ]
    
    rng = np.random.RandomState(42)
    
    for config in configs:
        if time.time() > deadline:
            break
            
        # Run local search with limited time
        # Allocate remaining time / remaining configs
        remaining_configs = len(configs) - configs.index(config)
        time_per_config = (deadline - time.time()) / max(1, remaining_configs)
        
        # Ensure at least some time
        time_per_config = max(1.0, time_per_config)
        
        optimized = _local_search(config, time_per_config, rng)
        
        score = _score(optimized)
        if score > best_score:
            best_score = score
            best_circles = optimized.copy()
            
    if best_circles is None:
        # Fallback
        best_circles = _make_initial_grid()
        best_circles[:, 2] = 0.01
        
    return best_circles

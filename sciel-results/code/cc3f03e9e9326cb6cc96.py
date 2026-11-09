import numpy as np
import time

def _is_valid_vectorized(centers, sides, tol=1e-7):
    """
    Vectorized validity check.
    centers: (N, 2)
    sides: (N,)
    """
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundary check
    if np.any(centers[:, 0] - sides/2 < -tol): return False
    if np.any(centers[:, 0] + sides/2 > 1 + tol): return False
    if np.any(centers[:, 1] - sides/2 < -tol): return False
    if np.any(centers[:, 1] + sides/2 > 1 + tol): return False
    
    # Overlap check
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    mask = ~np.eye(n, dtype=bool)
    
    if np.any((dx < sep) & (dy < sep) & mask):
        return False
        
    return True

def _initial_packing_grid():
    """
    Returns the known good packing with sum 3.0.
    2 squares of 0.5, 8 squares of 0.25.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # 2 Large at bottom
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    # 8 Small in top half (y in [0.5, 1.0])
    # Two rows of 4
    for k in range(8):
        row = k // 4  # 0 or 1
        col = k % 4   # 0 to 3
        cy = 0.625 + row * 0.25
        cx = 0.125 + col * 0.25
        centers[2+k] = [cx, cy]
        sides[2+k] = 0.25
        
    return centers, sides

def _random_valid_packing(rng):
    """
    Generate a random valid packing by placing squares one by one.
    This is slow but provides diverse starts.
    """
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Try to place 10 squares
    # Strategy: Random sizes, random positions, reject if invalid.
    # To make it faster, we can try smaller sizes first.
    
    # We'll use a simple greedy approach:
    # 1. Sort indices randomly.
    # 2. For each, pick a random size (biased towards smaller to fit more).
    # 3. Pick a random position.
    # 4. Check validity against placed squares and bounds.
    # 5. If invalid, retry with smaller size or different position.
    
    # Actually, for N=10, a simpler "shuffle and slide" or just using the grid 
    # with random perturbations is safer and faster for initialization.
    # Let's stick to perturbations of the known good grid for restarts, 
    # and maybe one truly random start.
    
    # For now, return a perturbed grid.
    c, s = _initial_packing_grid()
    # Perturb positions slightly
    c += rng.uniform(-0.05, 0.05, c.shape)
    # Perturb sizes slightly
    s += rng.uniform(-0.02, 0.02, s.shape)
    s = np.clip(s, 0.05, 1.0)
    
    # Clip centers to bounds
    for i in range(n):
        si = s[i]
        c[i, 0] = np.clip(c[i, 0], si/2, 1 - si/2)
        c[i, 1] = np.clip(c[i, 1], si/2, 1 - si/2)
        
    # It might be invalid. If so, we'll just let the SA fix it or reject it.
    # But for a "valid" initial state, we need to ensure it's valid.
    # The perturbation might break validity.
    # Let's just return the base grid for reliability, or a very small perturbation.
    # A better random start:
    # Place 10 squares of size 0.1 in a 10x1 grid? No, 10 squares of 0.1 fit in 1x1.
    # Sum = 1.0.
    # Let's just use the grid.
    
    return c, s

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 625 # Safety margin
    
    # Best solution found so far
    best_centers, best_sides = _initial_packing_grid()
    best_val = np.sum(best_sides)
    
    # SA Parameters
    T0 = 0.1
    T_min = 1e-6
    cooling_rate = 0.999
    max_iters_per_run = 50000
    stall_threshold = 5000 # No improvement for this many iters -> restart
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Initialize current solution
        # 50% chance to start from best, 50% from perturbed best
        if rng.random() < 0.5 and best_val > 3.0:
            centers = best_centers + rng.uniform(-0.02, 0.02, (n, 2))
            sides = best_sides + rng.uniform(-0.01, 0.01, n)
        else:
            centers, sides = _initial_packing_grid()
            centers = centers + rng.uniform(-0.05, 0.05, (n, 2))
            sides = sides + rng.uniform(-0.02, 0.02, n)
            
        # Ensure basic validity of start (clip to bounds)
        for i in range(n):
            s = max(0.01, min(1.0, sides[i]))
            sides[i] = s
            centers[i, 0] = np.clip(centers[i, 0], s/2, 1 - s/2)
            centers[i, 1] = np.clip(centers[i, 1], s/2, 1 - s/2)
            
        if not _is_valid_vectorized(centers, sides, tol):
            # If start is invalid, try to fix it or just start with grid
            centers, sides = _initial_packing_grid()
            
        current_val = np.sum(sides)
        T = T0
        last_improvement_iter = 0
        iter_count = 0
        
        while iter_count < max_iters_per_run and time.time() < deadline:
            iter_count += 1
            if iter_count % 1000 == 0 and time.time() > deadline:
                break
                
            # Generate neighbor
            new_centers = centers.copy()
            new_sides = sides.copy()
            
            idx = rng.integers(0, n)
            
            # Adaptive noise: larger when T is high
            noise_scale = 0.05 * (1 + T)
            
            new_centers[idx, 0] += rng.uniform(-noise_scale, noise_scale)
            new_centers[idx, 1] += rng.uniform(-noise_scale, noise_scale)
            new_sides[idx] += rng.uniform(-0.05, 0.05)
            
            # Enforce bounds
            s = new_sides[idx]
            if s <= 0.01:
                new_sides[idx] = 0.01
            elif s > 1.0:
                new_sides[idx] = 1.0
            else:
                new_sides[idx] = s
            
            s = new_sides[idx]
            new_centers[idx, 0] = np.clip(new_centers[idx, 0], s/2, 1 - s/2)
            new_centers[idx, 1] = np.clip(new_centers[idx, 1], s/2, 1 - s/2)
            
            # Check validity
            if _is_valid_vectorized(new_centers, new_sides, tol):
                new_val = np.sum(new_sides)
                delta = new_val - current_val
                
                # Acceptance
                if delta > 0 or rng.random() < np.exp(delta / T):
                    centers = new_centers
                    sides = new_sides
                    current_val = new_val
                    
                    if current_val > best_val:
                        best_val = current_val
                        best_centers = centers.copy()
                        best_sides = sides.copy()
                        last_improvement_iter = iter_count
                        
            T *= cooling_rate
            
            # Stall detection
            if iter_count - last_improvement_iter > stall_threshold:
                break
                
    return best_centers, best_sides

# EVOLVE-BLOCK-END

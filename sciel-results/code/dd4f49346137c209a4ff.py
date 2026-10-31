"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-6):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    # We use a slightly tighter tolerance for safety
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii, tol=1e-7):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(50):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + tol:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r + tol:
                    # Scale both radii to make them just touch with a small gap
                    if sum_r > 1e-9:
                        k = (dist - tol) / (sum_r + 1e-12)
                    else:
                        k = 0.99
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_greedy_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 5000 random points for better coverage
        for _ in range(5000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_hexagonal():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.12
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(5):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            # Clip to unit square
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            pos[k] = [x, y]
            radii[k] = r
            k += 1
            
    # If we have fewer than N, fill remaining with small circles in corners
    while k < N:
        if k == 25:
             pos[k] = [0.5, 0.5]
             radii[k] = 0.05
        else:
             pos[k] = [0.1, 0.1]
             radii[k] = 0.05
        k += 1

    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_anneal(pos, radii, steps=2000, start_temp=0.1, end_temp=0.001, rng=None):
    """
    Structural change: Force-directed annealing.
    Uses repulsive forces for overlaps and attractive forces for boundaries/center.
    Temperature allows accepting worse states to escape local minima.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Pre-compute pairs for efficiency
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for step in range(steps):
        # Geometric cooling
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Compute forces
        forces_pos = np.zeros((N, 2))
        forces_rad = np.zeros(N)
        
        # 1. Boundary forces (push back if outside, pull in if too close to edge? No, just push back)
        for i in range(N):
            # Left
            if pos[i, 0] < radii[i]:
                forces_pos[i, 0] += (radii[i] - pos[i, 0])
            # Right
            if pos[i, 0] > 1.0 - radii[i]:
                forces_pos[i, 0] -= (pos[i, 0] - (1.0 - radii[i]))
            # Bottom
            if pos[i, 1] < radii[i]:
                forces_pos[i, 1] += (radii[i] - pos[i, 1])
            # Top
            if pos[i, 1] > 1.0 - radii[i]:
                forces_pos[i, 1] -= (pos[i, 1] - (1.0 - radii[i]))
        
        # 2. Overlap forces (repulsive) and Radius growth forces (attractive to grow)
        for i, j in pairs:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = radii[i] + radii[j]
            
            if dist < sum_r:
                # Overlap: push apart
                if dist > 1e-6:
                    push = (sum_r - dist) / dist
                    forces_pos[i, 0] += dx * push
                    forces_pos[i, 1] += dy * push
                    forces_pos[j, 0] -= dx * push
                    forces_pos[j, 1] -= dy * push
                else:
                    # If coincident, random push
                    forces_pos[i, 0] += rng.uniform(-1, 1)
                    forces_pos[i, 1] += rng.uniform(-1, 1)
                    forces_pos[j, 0] += rng.uniform(-1, 1)
                    forces_pos[j, 1] += rng.uniform(-1, 1)
                
                # Shrink radii if overlapping significantly
                forces_rad[i] -= 0.5 * (sum_r - dist)
                forces_rad[j] -= 0.5 * (sum_r - dist)
            else:
                # Not overlapping: encourage growth
                # Force to grow radius is proportional to available space
                # Simple heuristic: grow if not touching
                gap = dist - sum_r
                if gap > 0:
                    forces_rad[i] += min(gap, 0.1)
                    forces_rad[j] += min(gap, 0.1)
        
        # 3. Apply moves with noise
        new_pos = pos + forces_pos * 0.1 + rng.normal(0, temp, (N, 2))
        new_radii = radii + forces_rad * 0.1 + rng.normal(0, temp * 0.5, N)
        
        # Clip to valid ranges
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, 1)
        new_radii = np.maximum(new_radii, 1e-5)
        new_radii = np.minimum(new_radii, 0.5)
        
        # Post-process to ensure hard validity for scoring
        new_pos_pp, new_radii_pp = _post_process(new_pos, new_radii, tol=1e-8)
        
        if _is_valid(new_pos_pp, new_radii_pp, tol=1e-7):
            new_score = np.sum(new_radii_pp)
            
            # Metropolis acceptance criterion
            delta = new_score - np.sum(radii)
            if delta > 0:
                # Accept if better
                pos = new_pos_pp
                radii = new_radii_pp
                if new_score > best_score:
                    best_score = new_score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            else:
                # Accept worse state with probability exp(delta / temp)
                if temp > 1e-9:
                    if rng.random() < np.exp(delta / temp):
                        pos = new_pos_pp
                        radii = new_radii_pp
        
        # Time check
        if time.time() > time.time() + 0: # Placeholder, actual check in construct
            pass
            
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # List of initializers to try
    initializers = [
        _initial_greedy_best_fit,
        _initial_hexagonal,
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
        lambda: _initial_random(4),
        lambda: _initial_random(5),
    ]
    
    # Try each initializer with Force-Directed Annealing
    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 15:
            break
            
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
            
        # Run Force-Directed Annealing
        try:
            pos_anneal, rad_anneal = _force_directed_anneal(pos_init, rad_init, steps=1000, start_temp=0.05, end_temp=0.001, rng=np.random.RandomState(idx))
            
            if _is_valid(pos_anneal, rad_anneal):
                score = np.sum(rad_anneal)
                if score > best_score:
                    best_score = score
                    best_pos = pos_anneal.copy()
                    best_radii = rad_anneal.copy()
        except Exception:
            pass

    # Final fallback if no valid result found
    if best_pos is None or not _is_valid(best_pos, best_radii):
        # Simple grid fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END

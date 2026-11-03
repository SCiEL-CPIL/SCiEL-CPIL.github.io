"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _eval_packing(centers, radii):
    """
    Evaluate the packing. Returns (sum_radii, is_valid).
    """
    n = len(radii)
    total_r = np.sum(radii)
    
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
            return -np.inf, False
            
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                return -np.inf, False
                
    return total_r, True

def _grid_init(n):
    """
    Initialize with a grid layout.
    """
    side = int(np.ceil(np.sqrt(n)))
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        gx = i % side
        gy = i // side
        # Center of cell
        centers[i][0] = (gx + 0.5) / side
        centers[i][1] = (gy + 0.5) / side
        # Max radius is half the cell size (distance to edge), minus small margin
        # Actually, distance to nearest edge is 0.5/side.
        radii[i] = 0.5 / side - 0.01
    return centers, radii

def _best_fit_local_search(centers, radii, n, max_iters=10000):
    """
    Improve the packing by moving circles to the position that allows
    the largest possible radius, given current constraints.
    This is a heuristic local search.
    """
    best_score, _ = _eval_packing(centers, radii)
    
    # We will try to increase radii by moving circles.
    # Strategy: Pick a random circle, try to move it to a better spot 
    # (one that allows a larger radius or maintains radius with less overlap).
    
    # To make this efficient, we can use a "best fit" approach:
    # For each circle, find the position in a local neighborhood that maximizes 
    # the minimum distance to other circles and boundaries.
    
    # However, a simpler and effective SA-like approach is to perturb positions 
    # and radii.
    
    # Let's implement a simple local search that tries to expand radii.
    # If a circle is blocked, we try to move it slightly to open space.
    
    current_score = best_score
    if not np.isfinite(current_score):
        # If invalid, try to fix it first
        # For now, assume input is valid from grid_init
        pass

    for _ in range(max_iters):
        # Pick a random circle
        i = np.random.randint(n)
        
        # Try to increase its radius slightly
        delta_r = 0.001
        new_r = radii[i] + delta_r
        
        # Check if it fits at current position
        x, y = centers[i]
        valid_pos = (x >= new_r and x <= 1.0 - new_r and y >= new_r and y <= 1.0 - new_r)
        
        if valid_pos:
            # Check overlap with others
            overlap = False
            for j in range(n):
                if i == j: continue
                dx = x - centers[j][0]
                dy = y - centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < new_r + radii[j]:
                    overlap = True
                    break
            
            if not overlap:
                radii[i] = new_r
                # Update score
                new_score, _ = _eval_packing(centers, radii)
                if new_score > current_score:
                    current_score = new_score
                else:
                    radii[i] -= delta_r
            else:
                # Try to move the circle to resolve overlap
                # Simple random walk in position
                dx_move = np.random.normal(0, 0.01)
                dy_move = np.random.normal(0, 0.01)
                new_x = x + dx_move
                new_y = y + dy_move
                
                # Check if new position is valid for current radius
                if (new_x >= radii[i] and new_x <= 1.0 - radii[i] and 
                    new_y >= radii[i] and new_y <= 1.0 - radii[i]):
                    
                    # Check overlap
                    valid_move = True
                    for j in range(n):
                        if i == j: continue
                        ddx = new_x - centers[j][0]
                        ddy = new_y - centers[j][1]
                        dist = np.sqrt(ddx*ddx + ddy*ddy)
                        if dist < radii[i] + radii[j]:
                            valid_move = False
                            break
                    
                    if valid_move:
                        centers[i] = [new_x, new_y]
        else:
            # Radius too big for position, shrink it? Or move it?
            # Shrink slightly to allow movement
            radii[i] = max(0.01, radii[i] * 0.95)

    return centers, radii

def _sa_packing(n, time_limit=10.0, seed=None):
    """
    Simulated Annealing for circle packing.
    State: centers (n,2), radii (n,)
    Objective: Maximize sum(radii) subject to non-overlap and boundary constraints.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Initialize
    centers, radii = _grid_init(n)
    
    # Ensure initial validity (grid init should be valid)
    current_score, is_valid = _eval_packing(centers, radii)
    if not is_valid:
        # Fallback: random small circles
        centers = np.random.rand(n, 2) * 0.8 + 0.1
        radii = np.full(n, 0.01)
        current_score, is_valid = _eval_packing(centers, radii)
        
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    T = 0.1
    T_min = 1e-4
    alpha = 0.995
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Propose a move
        move_type = random.random()
        
        new_centers = centers.copy()
        new_radii = radii.copy()
        
        if move_type < 0.5:
            # Move a circle
            i = np.random.randint(n)
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            new_centers[i][0] += dx
            new_centers[i][1] += dy
        elif move_type < 0.8:
            # Change radius of a circle
            i = np.random.randint(n)
            delta_r = np.random.normal(0, 0.01)
            new_radii[i] = max(0.01, new_radii[i] + delta_r)
        else:
            # Perturb multiple circles slightly (cluster move)
            for _ in range(3):
                i = np.random.randint(n)
                dx = np.random.normal(0, 0.02)
                dy = np.random.normal(0, 0.02)
                new_centers[i][0] += dx
                new_centers[i][1] += dy
        
        # Evaluate new state
        # We need a fast validity check.
        # If invalid, penalty is huge.
        new_score, is_valid = _eval_packing(new_centers, new_radii)
        
        if not is_valid:
            # Assign a very low score based on violation magnitude?
            # For simplicity, reject if invalid, or accept with high prob if close.
            # Let's calculate a penalty score for invalid states to allow "tunneling"
            # Penalty = sum of overlaps + boundary violations
            penalty = 0.0
            for i in range(n):
                x, y = new_centers[i]
                r = new_radii[i]
                if x < r: penalty += (r - x)**2
                if x > 1.0 - r: penalty += (x - (1.0 - r))**2
                if y < r: penalty += (r - y)**2
                if y > 1.0 - r: penalty += (y - (1.0 - r))**2
                
            for i in range(n):
                for j in range(i+1, n):
                    dx = new_centers[i][0] - new_centers[j][0]
                    dy = new_centers[i][1] - new_centers[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = new_radii[i] + new_radii[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            new_score = -penalty * 10.0 - np.sum(new_radii)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            centers = new_centers
            radii = new_radii
            current_score = new_score
            
            if current_score > best_score and is_valid:
                best_score = current_score
                best_centers = centers.copy()
                best_radii = radii.copy()
        
        T *= alpha
        
    return best_centers, best_radii

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    best_centers = None
    best_radii = None
    best_score = -np.inf
    
    # Try a few different SA runs with different seeds
    # and different initializations if time permits
    
    for trial in range(5):
        if time.time() > deadline:
            break
            
        # Use different seeds for diversity
        seed = trial + 100
        
        # For trial 0, use grid init. For others, maybe random init?
        # Let's stick to grid init for SA as it's a good starting point.
        # The SA will handle the optimization.
        
        c, r = _sa_packing(N, time_limit=20.0, seed=seed)
        
        score, valid = _eval_packing(c, r)
        
        if valid and score > best_score:
            best_score = score
            best_centers = c
            best_radii = r
            
        # If we have time, run a local polish on the best found so far
        if best_centers is not None and time.time() < deadline - 5:
             # Simple local polish: try to increase radii of the best solution
             c_polished, r_polished = _best_fit_local_search(best_centers.copy(), best_radii.copy(), N, max_iters=5000)
             score_polished, valid_polished = _eval_packing(c_polished, r_polished)
             if valid_polished and score_polished > best_score:
                 best_score = score_polished
                 best_centers = c_polished
                 best_radii = r_polished

    if best_centers is None:
        # Fallback to grid
        c, r = _grid_init(N)
        best_centers = c
        best_radii = r
        
    # Final validation and cleanup
    result = np.column_stack([best_centers, best_radii])
    
    # Ensure strict validity: clip radii to fit boundaries
    for i in range(N):
        x, y, ri = result[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if ri > max_r:
            ri = max_r
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ry) if False else np.clip(y, ri, 1.0 - ri)
        
    # Check for overlaps one last time and shrink if necessary
    # This is a safety net. If SA left overlaps, we shrink.
    for _ in range(10):
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result[i, 2] + result[j, 2]
                if dist < min_dist:
                    # Overlap found, shrink the smaller one
                    overlap = min_dist - dist
                    if result[i, 2] < result[j, 2]:
                        result[i, 2] = max(0.001, result[i, 2] - overlap/2)
                    else:
                        result[j, 2] = max(0.001, result[j, 2] - overlap/2)
                    modified = True
        if not modified:
            break
            
    return result
# EVOLVE-BLOCK-END

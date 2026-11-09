"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_score(circles):
    """Compute the sum of radii if valid, else 0."""
    n = len(circles)
    if n == 0:
        return 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return 0.0
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return 0.0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return 0.0
    return float(np.sum(circles[:, 2]))

def _max_radius(i, x, y, circles):
    """Calculate the maximum radius for circle i at position (x,y)."""
    n = len(circles)
    # Boundary constraints
    r_bound = min(x, 1.0 - x, y, 1.0 - y)
    if r_bound <= 0:
        return 0.0
    
    r_max = r_bound
    # Neighbor constraints
    for j in range(n):
        if i == j:
            continue
        xj, yj, rj = circles[j]
        dist = np.sqrt((x - xj)**2 + (y - yj)**2)
        # The new circle of radius r at (x,y) must satisfy dist >= r + rj
        # So r <= dist - rj
        allowed = dist - rj
        if allowed < r_max:
            r_max = allowed
            
    if r_max < 0:
        return 0.0
    return r_max

def _make_initial_grid(n):
    """Create an initial grid layout of n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Start with small radius to ensure no overlap
            r = 0.4 / side 
            cells.append([x, y, r])
    return np.array(cells)

def construct():
    n = N
    deadline = time.time() + 140.0  # Safety margin
    
    # Initialize
    circles = _make_initial_grid(n)
    
    # Initial pass to set radii to max possible
    for i in range(n):
        x, y, _ = circles[i]
        circles[i, 2] = _max_radius(i, x, y, circles)
        
    best_score = _compute_score(circles)
    best_circles = circles.copy()
    
    # Local Search Loop
    # We alternate between trying to grow radii and moving circles to allow growth
    stuck_count = 0
    last_improvement_time = time.time()
    
    while time.time() < deadline:
        improved = False
        
        # Shuffle order to avoid bias
        indices = np.random.permutation(n)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            xi, yi, ri = circles[i]
            
            # 1. Try to grow current radius
            r_max = _max_radius(i, xi, yi, circles)
            if r_max > ri + 1e-7:
                circles[i, 2] = r_max
                improved = True
                continue
            
            # 2. If cannot grow, try to move to a better position
            # Generate candidate moves
            candidates = []
            
            # Small random jitter
            for _ in range(5):
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)
                nx, ny = xi + dx, yi + dy
                if 0 <= nx <= 1 and 0 <= ny <= 1:
                    candidates.append((nx, ny))
            
            # Move towards center of mass of empty space? 
            # Heuristic: Move away from the nearest neighbor
            nearest_dist = float('inf')
            nearest_idx = -1
            for j in range(n):
                if i == j: continue
                xj, yj, _ = circles[j]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = j
            
            if nearest_idx != -1:
                xj, yj, _ = circles[nearest_idx]
                # Direction away from neighbor
                dx = xi - xj
                dy = yi - yj
                norm = np.sqrt(dx*dx + dy*dy)
                if norm > 1e-9:
                    dx /= norm
                    dy /= norm
                    # Try moving away
                    for step in [0.02, 0.05, 0.1]:
                        nx, ny = xi + dx*step, yi + dy*step
                        if 0 <= nx <= 1 and 0 <= ny <= 1:
                            candidates.append((nx, ny))
            
            # Try candidates
            best_cand_r = ri
            best_cand_pos = None
            for cx, cy in candidates:
                r_cand = _max_radius(i, cx, cy, circles)
                if r_cand > best_cand_r + 1e-7:
                    best_cand_r = r_cand
                    best_cand_pos = (cx, cy)
            
            if best_cand_pos is not None:
                circles[i, 0] = best_cand_pos[0]
                circles[i, 1] = best_cand_pos[1]
                circles[i, 2] = best_cand_r
                improved = True
        
        if improved:
            stuck_count = 0
            # Check score
            current_score = _compute_score(circles)
            if current_score > best_score:
                best_score = current_score
                best_circles = circles.copy()
                last_improvement_time = time.time()
        else:
            stuck_count += 1
            # If stuck, perform a shake
            if stuck_count > 50:
                # Randomly perturb 2-3 circles
                num_shake = random.randint(2, 4)
                shake_indices = np.random.choice(n, num_shake, replace=False)
                for idx in shake_indices:
                    # Move to random position
                    nx = np.random.uniform(0.1, 0.9)
                    ny = np.random.uniform(0.1, 0.9)
                    circles[idx, 0] = nx
                    circles[idx, 1] = ny
                    # Reset radius to max possible there (might be small)
                    circles[idx, 2] = _max_radius(idx, nx, ny, circles)
                stuck_count = 0
                # Re-optimize radii for all after shake
                for i in range(n):
                    x, y, _ = circles[i]
                    circles[i, 2] = _max_radius(i, x, y, circles)
                
                # If we've been stuck for a long time and score is bad, consider restarting?
                # For now, just shake.

    # Final validation
    final_score = _compute_score(best_circles)
    if final_score <= 0:
        # Fallback
        best_circles = _make_initial_grid(n)
        for i in range(n):
            x, y, _ = best_circles[i]
            best_circles[i, 2] = _max_radius(i, x, y, best_circles)
            
    return best_circles
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 26

# EVOLVE-BLOCK-START
def _max_radius_at(pos, others):
    """Calculate the maximum radius a circle at 'pos' can have without overlapping others or boundaries."""
    x, y = pos
    r = min(x, y, 1.0 - x, 1.0 - y)
    if r <= 0:
        return 0.0
    for (ox, oy, orr) in others:
        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
        # If circles are already overlapping, the max radius is effectively 0 (or negative in math, but we clamp to 0)
        # However, in LS, we usually start with small radii and expand.
        # The constraint is: r <= dist - orr
        max_r_constraint = dist - orr
        if max_r_constraint < r:
            r = max_r_constraint
    if r < 0:
        return 0.0
    return r

def _ls_step(circles, time_step):
    """
    Perform one Lubachevsky-Steampole step.
    1. Find the circle with the largest possible radius expansion (max clearance).
    2. Expand that circle by time_step.
    3. Resolve any collisions created by this expansion by pushing other circles away.
    """
    n = len(circles)
    if n == 0:
        return circles
    
    # Copy circles to avoid mutation during search
    c = [list(circ) for circ in circles]
    
    # 1. Identify the circle to expand
    # We want to maximize the sum of radii. The LS algorithm typically picks the circle 
    # that can be expanded the most relative to its current radius, or simply the one 
    # with the most free space. 
    # A common heuristic is to pick the circle with the largest 'clearance' 
    # (max_radius - current_radius).
    
    best_idx = -1
    best_gain = -1.0
    
    for i in range(n):
        x, y, r = c[i]
        others = [c[j] for j in range(n) if j != i]
        max_r = _max_radius_at((x, y), others)
        gain = max_r - r
        if gain > best_gain:
            best_gain = gain
            best_idx = i
            
    if best_idx == -1 or best_gain <= 0:
        # No room to expand? This shouldn't happen if we start small enough, 
        # but if it does, we might need to jiggle or terminate.
        # For now, just return current state.
        return np.array(c)
        
    # 2. Expand the best circle
    x, y, r = c[best_idx]
    c[best_idx][2] = r + time_step
    
    # 3. Resolve collisions
    # The expanded circle might now overlap with others.
    # We need to push the other circles away.
    # Iterate until no more overlaps or max iterations reached.
    # Simple approach: For each overlapping pair, push the smaller circle (or the one not being expanded) away.
    # To be more robust, we can do a few iterations of collision resolution.
    
    for _ in range(10): # Max iterations for collision resolution
        moved = False
        for i in range(n):
            if i == best_idx:
                continue
            xi, yi, ri = c[i]
            xj, yj, rj = c[best_idx]
            
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            min_dist = ri + rj
            
            if dist < min_dist - 1e-9:
                # Collision!
                if dist < 1e-9:
                    # Perfect overlap, push in random direction
                    angle = random.uniform(0, 2 * np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                else:
                    dx = (xi - xj) / dist
                    dy = (yi - yj) / dist
                
                # Push circle i away from best_idx
                # We want to increase distance to min_dist.
                # Current distance is dist. We need to move i by (min_dist - dist) in direction (dx, dy).
                # But we also need to respect boundaries.
                
                overlap = min_dist - dist
                # To be safe and smooth, we can push by a fraction of the overlap or the full amount.
                # Full amount is standard for hard constraints.
                c[i][0] += dx * overlap
                c[i][1] += dy * overlap
                moved = True
                
        if not moved:
            break
            
    # 4. Enforce boundary constraints for ALL circles
    # The push might have pushed some circles outside [0,1]^2.
    for i in range(n):
        x, y, r = c[i]
        # Clamp x
        if x < r:
            c[i][0] = r
        elif x > 1.0 - r:
            c[i][0] = 1.0 - r
        # Clamp y
        if y < r:
            c[i][1] = r
        elif y > 1.0 - r:
            c[i][1] = 1.0 - r
            
    # Note: Boundary clamping might create new overlaps. 
    # A more rigorous LS implementation would handle this, but for this heuristic, 
    # we rely on the next step's collision detection to fix any residual issues 
    # or we assume the push was gentle enough. 
    # To improve stability, we can re-check overlaps and do a simple gradient step for overlaps.
    
    # Simple overlap resolution for any remaining overlaps (due to boundary clamping)
    for _ in range(5):
        any_overlap = False
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = c[i]
                xj, yj, rj = c[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_dist = ri + rj
                if dist < min_dist - 1e-9:
                    any_overlap = True
                    if dist < 1e-9:
                        angle = random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                    else:
                        dx = (xi - xj) / dist
                        dy = (yi - yj) / dist
                    overlap = min_dist - dist
                    # Push both circles apart by half the overlap
                    c[i][0] += dx * overlap * 0.5
                    c[i][1] += dy * overlap * 0.5
                    c[j][0] -= dx * overlap * 0.5
                    c[j][1] -= dy * overlap * 0.5
        if not any_overlap:
            break
            
    # Final boundary clamp
    for i in range(n):
        x, y, r = c[i]
        if x < r: c[i][0] = r
        elif x > 1.0 - r: c[i][0] = 1.0 - r
        if y < r: c[i][1] = r
        elif y > 1.0 - r: c[i][1] = 1.0 - r

    return np.array(c)

def _run_ls(num_circles, time_step, max_steps, seed=0):
    """Run the LS algorithm for a given number of circles."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize circles
    # Start with very small circles in a grid to avoid initial overlaps
    side = int(np.ceil(np.sqrt(num_circles)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= num_circles:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.01 # Small initial radius
            cells.append([x, y, r])
            
    # Add some random jitter to break symmetry
    for i in range(len(cells)):
        cells[i][0] += np.random.uniform(-0.01, 0.01)
        cells[i][1] += np.random.uniform(-0.01, 0.01)
        
    circles = np.array(cells[:num_circles])
    
    for step in range(max_steps):
        circles = _ls_step(circles, time_step)
        
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s margin for safety
    
    best_circles = None
    best_score = -1.0
    
    # Try multiple seeds and configurations
    # The LS algorithm is sensitive to initialization and time step.
    # We'll try a few different time steps and seeds.
    
    configs = [
        (26, 0.005, 2000, 1),
        (26, 0.002, 5000, 2),
        (26, 0.01, 1000, 3),
        (26, 0.001, 10000, 4),
        (26, 0.005, 2000, 5),
        (26, 0.003, 3000, 6),
    ]
    
    for n_circles, t_step, max_steps, seed in configs:
        if time.time() > deadline:
            break
            
        try:
            circles = _run_ls(n_circles, t_step, max_steps, seed)
            
            # Validate and score
            # Check if circles are valid (inside square, no overlaps)
            valid = True
            for i in range(len(circles)):
                x, y, r = circles[i]
                if x < -1e-6 or x > 1 + 1e-6 or y < -1e-6 or y > 1 + 1e-6 or r < -1e-6:
                    valid = False
                    break
                if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
                    valid = False
                    break
                    
            if valid:
                for i in range(len(circles)):
                    for j in range(i+1, len(circles)):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                        if dist < ri + rj - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
            if valid:
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
                    
        except Exception:
            continue
            
    # If no valid circles found (should not happen), fallback to grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        
    return best_circles
# EVOLVE-BLOCK-END

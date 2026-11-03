import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Parameters for Simulated Annealing
    T_initial = 1.0
    T_min = 1e-6
    cooling_rate = 0.9995
    max_iter = 200000
    move_prob_pos = 0.5
    move_prob_rad = 0.5
    step_size_pos = 0.05
    step_size_rad = 0.01
    
    # Objective: Maximize sum of radii. 
    # We want to minimize -sum(r).
    def get_energy(x, y, r):
        return -np.sum(r)

    def is_valid(x, y, r, eps=1e-6):
        # Check boundaries
        if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
            return False
        if np.any(r < 0):
            return False
        # Check overlaps
        # For efficiency, we can use a simple loop or vectorized check. 
        # With N=26, O(N^2) is fast.
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                r_sum = r[i] + r[j]
                if dist_sq < (r_sum - eps)**2:
                    return False
        return True

    def project_constraints(x, y, r):
        """
        Adjusts x, y, r to satisfy constraints.
        Returns adjusted x, y, r.
        """
        # 1. Boundary projection
        # If x < r, set x = r. If x > 1-r, set x = 1-r.
        # This might cause overlaps, so we iterate or handle overlaps next.
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        # 2. Overlap resolution
        # We perform a few iterations of pushing apart overlapping circles.
        # This is a heuristic projection.
        for _ in range(10):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    r_sum = r[i] + r[j]
                    if dist < r_sum and dist > 1e-9:
                        # Push apart
                        overlap = r_sum - dist
                        # Move each by half the overlap? Or move the smaller one more?
                        # Let's move both.
                        nx = dx / dist
                        ny = dy / dist
                        shift = overlap / 2.0 + 1e-5
                        x[i] += nx * shift
                        y[i] += ny * shift
                        x[j] -= nx * shift
                        y[j] -= ny * shift
                        moved = True
                    elif dist <= 1e-9:
                        # Coincident centers, random push
                        angle = random.uniform(0, 2*np.pi)
                        shift = r_sum / 2.0 + 1e-5
                        x[i] += np.cos(angle) * shift
                        y[i] += np.sin(angle) * shift
                        x[j] -= np.cos(angle) * shift
                        y[j] -= np.sin(angle) * shift
                        moved = True
            
            # Re-apply boundary constraints after moving
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            if not moved:
                break
                
        # 3. If boundaries are still violated (e.g., r > 0.5), shrink radius?
        # For N=26, max r is roughly 0.5/sqrt(26) ~ 0.1 if grid, but hex packing allows larger.
        # Actually, if x is clipped to [r, 1-r], and r is large, x might be forced to a point.
        # If the interval [r, 1-r] is empty (r > 0.5), we must shrink r.
        # But r > 0.5 is impossible for N>1.
        # However, if x is pushed to a corner and r is large, it might stick out?
        # No, clipping ensures x >= r and x <= 1-r. So x is inside [r, 1-r].
        # This guarantees x-r >= 0 and 1-x-r >= 0. So boundaries are satisfied.
        
        return x, y, r

    # Initialize
    # Start with a grid or random
    random.seed(42)
    np.random.seed(42)
    
    # Grid initialization
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # 5x5 grid + 1
    # Let's use a slightly perturbed grid
    side = 5
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx < N:
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                r[idx] = 0.5 / side * 0.9 # Slightly smaller than cell half-width
                idx += 1
    if idx < N:
        # Place last one randomly
        x[idx] = random.uniform(0.1, 0.9)
        y[idx] = random.uniform(0.1, 0.9)
        r[idx] = 0.05

    # Initial projection
    x, y, r = project_constraints(x, y, r)
    
    current_energy = get_energy(x, y, r)
    best_energy = current_energy
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    T = T_initial
    
    for i_iter in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        # Generate move
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if random.random() < move_prob_pos:
            # Move a random circle
            idx = random.randint(0, N-1)
            dx = random.uniform(-step_size_pos, step_size_pos)
            dy = random.uniform(-step_size_pos, step_size_pos)
            new_x[idx] += dx
            new_y[idx] += dy
        else:
            # Grow/Shrink a random circle
            idx = random.randint(0, N-1)
            dr = random.uniform(-step_size_rad, step_size_rad)
            new_r[idx] = max(1e-4, new_r[idx] + dr)
            
        # Project to feasible space
        new_x, new_y, new_r = project_constraints(new_x, new_y, new_r)
        
        new_energy = get_energy(new_x, new_y, new_r)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0:
            accept = True
        else:
            # Boltzmann probability
            # If T is very small, exp(delta_e/T) will be 0 if delta_e > 0
            if T > 1e-9:
                prob = np.exp(-delta_e / T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            x, y, r = new_x, new_y, new_r
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        T *= cooling_rate
        
    # Final check and return
    # Ensure the best solution is valid
    best_x, best_y, best_r = project_constraints(best_x, best_y, best_r)
    
    # If the SA result is worse than a simple grid fallback, use fallback?
    # Grid fallback sum of radii: 26 * (0.5/5 * 0.9) = 26 * 0.09 = 2.34
    # SA should do better.
    
    result = np.column_stack((best_x, best_y, best_r))
    
    # One last validation check
    if not is_valid(best_x, best_y, best_r):
         # Fallback to grid if SA failed to converge to a valid state
         # This shouldn't happen with projection, but just in case
         side = 5
         cells = []
         for i in range(side):
             for j in range(side):
                 if len(cells) < N:
                     cells.append([(i+0.5)/side, (j+0.5)/side, 0.5/side * 0.9])
         if len(cells) < N:
             cells.append([0.5, 0.5, 0.05]) # Might overlap, but valid if small enough?
             # Actually, just return the grid we started with
         return np.array(cells)
         
    return result

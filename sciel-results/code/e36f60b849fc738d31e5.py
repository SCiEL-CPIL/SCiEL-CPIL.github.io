import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def compute_score_and_forces(circles):
        """Compute score and repulsive forces for local optimization."""
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check validity
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-10 or x[i] > 1 - r[i] + 1e-10:
                valid = False
            if y[i] < r[i] - 1e-10 or y[i] > 1 - r[i] + 1e-10:
                valid = False
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if not valid:
            return -np.inf, None
        
        score = np.sum(r)
        
        # Compute forces: repulsion between overlapping circles, attraction to expand
        forces_x = np.zeros(N)
        forces_y = np.zeros(N)
        forces_r = np.zeros(N)
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i] + 0.01:
                forces_x[i] += 1.0
            if x[i] > 1 - r[i] - 0.01:
                forces_x[i] -= 1.0
            if y[i] < r[i] + 0.01:
                forces_y[i] += 1.0
            if y[i] > 1 - r[i] - 0.01:
                forces_y[i] -= 1.0
        
        # Pairwise repulsion for close circles
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist * 1.5:  # Only interact if relatively close
                    if dist > 1e-10:
                        # Repulsive force
                        strength = (min_dist - dist) / min_dist
                        fx = strength * dx / dist
                        fy = strength * dy / dist
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
        
        # Want to increase radii
        forces_r[:] = 0.5
        
        return score, (forces_x, forces_y, forces_r)
    
    def optimize(circles, max_iter=1000):
        """Local optimization using force-directed approach."""
        current = circles.copy()
        current_score = compute_score_and_forces(current)[0]
        
        # If invalid, shrink radii
        if current_score == -np.inf:
            for i in range(N):
                current[i, 2] *= 0.9
            current_score = compute_score_and_forces(current)[0]
        
        step_size = 0.01
        best_in_run = current.copy()
        best_score_in_run = current_score
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            score, forces = compute_score_and_forces(current)
            
            if forces is None:
                # Invalid, try to fix
                current[:, 2] *= 0.95
                for i in range(N):
                    current[i, 0] = np.clip(current[i, 0], current[i, 2], 1 - current[i, 2])
                    current[i, 1] = np.clip(current[i, 1], current[i, 2], 1 - current[i, 2])
                continue
            
            fx, fy, fr = forces
            
            # Update positions and radii
            current[:, 0] += step_size * fx
            current[:, 1] += step_size * fy
            current[:, 2] += step_size * fr
            
            # Clamp to valid range
            for i in range(N):
                r = current[i, 2]
                current[i, 2] = np.clip(r, 1e-6, 0.4)
                current[i, 0] = np.clip(current[i, 0], current[i, 2], 1 - current[i, 2])
                current[i, 1] = np.clip(current[i, 1], current[i, 2], 1 - current[i, 2])
            
            # Check if improved
            new_score = compute_score_and_forces(current)[0]
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score_in_run:
                    best_score_in_run = new_score
                    best_in_run = current.copy()
            else:
                # Revert and reduce step size
                current = current - step_size * np.array([fx, fy, fr]).T
                step_size *= 0.9
        
        return best_in_run, best_score_in_run
    
    def grid_init(jitter_scale=0.1):
        """Grid-based initialization with jitter."""
        side = int(np.ceil(np.sqrt(N)))
        circles = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                r = 0.5 / side * 0.85
                x = (i + 0.5) / side + np.random.uniform(-jitter_scale, jitter_scale) / side
                y = (j + 0.5) / side + np.random.uniform(-jitter_scale, jitter_scale) / side
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)
                circles[count] = [x, y, r]
                count += 1
        return circles
    
    best_circles = None
    best_score = -np.inf
    
    np.random.seed(42)
    
    # Multiple restarts with different initializations
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart % 2 == 0:
            init = grid_init(jitter_scale=0.05 + 0.05 * restart)
        else:
            # Random initialization
            init = np.zeros((N, 3))
            for i in range(N):
                r = np.random.uniform(0.03, 0.12)
                x = np.random.uniform(r, 1 - r)
                y = np.random.uniform(r, 1 - r)
                init[i] = [x, y, r]
        
        result, sc = optimize(init, max_iter=500)
        
        if sc > best_score:
            best_score = sc
            best_circles = result.copy()
    
    # Fallback
    if best_circles is None or best_score == -np.inf:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    # Final cleanup
    for i in range(N):
        r = best_circles[i, 2]
        best_circles[i, 0] = np.clip(best_circles[i, 0], r, 1 - r)
        best_circles[i, 1] = np.clip(best_circles[i, 1], r, 1 - r)
        best_circles[i, 2] = max(1e-6, r)
    
    return best_circles

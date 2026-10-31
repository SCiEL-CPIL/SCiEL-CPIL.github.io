import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def get_score(s):
        # s is (N, 3)
        x = s[:, 0]
        y = s[:, 1]
        r = s[:, 2]
        
        # Boundary violations
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        bound_viol = np.sum(np.maximum(0, r - max_r_bound))
        
        # Pairwise violations
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        pair_viol = np.sum(np.maximum(0, sum_r - dist))
        
        total_viol = bound_viol + pair_viol
        return np.sum(r) - 10.0 * total_viol, total_viol

    def generate_radial_init():
        pts = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Center
        pts[0] = [0.5, 0.5]
        
        # Ring 1: 8 points
        r1 = 0.2
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for i in range(8):
            pts[i+1] = [0.5 + r1*np.cos(angles1[i]), 0.5 + r1*np.sin(angles1[i])]
            
        # Ring 2: 17 points
        r2 = 0.42
        angles2 = np.linspace(0, 2*np.pi, 17, endpoint=False) + np.random.uniform(-0.1, 0.1, 17)
        for i in range(17):
            pts[i+9] = [0.5 + r2*np.cos(angles2[i]), 0.5 + r2*np.sin(angles2[i])]
            
        radii[:] = 0.01
        return np.hstack([pts, radii[:, None]])

    def generate_grid_init():
        # 5x5 grid with one removed, or 6x5? 26 is close to 5x5=25.
        # Let's do a 5x5 grid plus one extra in the middle? No, 26 distinct.
        # Maybe 6x5 grid = 30, remove 4?
        # Or just random grid perturbation.
        # Let's try a 5x5 grid (25 circles) + 1 circle in a gap?
        # Simpler: 6x5 grid (30 points), pick 26 random ones?
        # Or 5x5 grid (25) + 1 at (0.5, 0.5) is already there?
        # Let's do 5x5 grid for first 25, and 1 extra near a corner?
        
        # Actually, a 6x5 grid is 30. 26 is 30-4.
        # Let's just use a 5x5 grid (25) and add one more in the center? No, center is occupied.
        # Let's use a 5x5 grid with spacing 0.18?
        # Spacing 0.18 -> 5 points fit in 0.72 width.
        # Let's just generate 26 points on a grid with some jitter.
        
        rows = 5
        cols = 5
        # 25 points
        pts = np.zeros((N, 2))
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N: break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                # Add jitter
                x += np.random.uniform(-0.02, 0.02)
                y += np.random.uniform(-0.02, 0.02)
                pts[idx] = [x, y]
                idx += 1
        
        # Add 26th point
        # Find a random spot? Or just put it at (0.5, 0.5) and shift others?
        # Let's just put it at (0.5, 0.5) and hope the optimizer moves it.
        # Or better: use 6x5 grid = 30 points, pick 26.
        # Let's restart grid init.
        
        rows = 5
        cols = 5
        # 25 points
        pts = np.zeros((N, 2))
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N: break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                pts[idx] = [x, y]
                idx += 1
        
        # 26th point: put it in the middle of the square?
        # The center of the 5x5 grid is (0.5, 0.5).
        # So we have a point at (0.5, 0.5).
        # Let's put the 26th point at (0.5, 0.5) and shift the others?
        # No, just put it at (0.5, 0.5) and let the optimizer handle the overlap.
        # Or better: 26 points in a 6x5 grid?
        # 6 rows, 5 cols = 30.
        # Let's pick 26 from a 6x5 grid.
        
        pts = np.zeros((N, 2))
        candidates = []
        for i in range(6):
            for j in range(5):
                x = (j + 0.5) / 5
                y = (i + 0.5) / 6
                candidates.append([x, y])
        
        # Pick 26 random candidates
        selected = random.sample(candidates, N)
        pts[:] = selected
        
        radii = np.zeros(N)
        radii[:] = 0.01
        return np.hstack([pts, radii[:, None]])

    best_state = None
    best_score = -np.inf
    
    # Multi-start local search
    # We will run multiple independent local searches.
    # Each search uses a different initialization.
    
    num_starts = 10
    time_per_start = (deadline - time.time()) / num_starts
    
    for s in range(num_starts):
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        # Choose init
        if s % 2 == 0:
            state = generate_radial_init()
        else:
            state = generate_grid_init()
        
        current_score, _ = get_score(state)
        
        # Local search: Simulated Annealing / Hill Climbing
        T = 0.05
        cooling = 0.999
        local_deadline = time.time() + time_per_start
        
        while time.time() < local_deadline:
            k = np.random.randint(N)
            move_type = np.random.randint(3)
            
            new_state = state.copy()
            step = T * 0.1
            
            if move_type == 0:
                new_state[k, 0] += np.random.normal(0, step)
            elif move_type == 1:
                new_state[k, 1] += np.random.normal(0, step)
            else:
                new_state[k, 2] += np.random.normal(0, step)
            
            new_state[k, 2] = max(0.001, new_state[k, 2])
            
            new_score, new_viol = get_score(new_state)
            
            if new_viol < 1e-6: # Valid
                if new_score > current_score:
                    state = new_state
                    current_score = new_score
                else:
                    delta = new_score - current_score
                    if np.random.random() < np.exp(delta / T):
                        state = new_state
                        current_score = new_score
            else:
                # Invalid move. Reject.
                pass
            
            T *= cooling
            if T < 0.001:
                T = 0.001
        
        if current_score > best_score:
            best_score = current_score
            best_state = state.copy()
    
    # Final check: ensure best_state is valid
    if best_state is None:
        # Fallback to a simple valid configuration
        best_state = np.zeros((N, 3))
        for i in range(N):
            best_state[i] = [0.5, 0.5, 0.01]
        # This will be invalid due to overlaps, so we need a valid fallback.
        # Let's just use the grid init with small radii.
        state = generate_grid_init()
        # Ensure validity by reducing radii
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, max_r_bound)
        # Check overlaps
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        if np.any(sum_r > dist + 1e-9):
            # Scale down
            scale = 0.5
            while True:
                r_check = r * scale
                sum_r_check = r_check[i_idx] + r_check[j_idx]
                if np.all(sum_r_check <= dist + 1e-9):
                    break
                scale *= 0.9
            r = r_check
        best_state = np.hstack([x[:, None], y[:, None], r[:, None]])
    
    # Verify best_state validity one last time
    x = best_state[:, 0]
    y = best_state[:, 1]
    r = best_state[:, 2]
    
    # Check bounds
    max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    if np.any(r > max_r_bound + 1e-9):
        # Scale down
        scale = 1.0
        for _ in range(10):
            r_check = r * scale
            max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
            if np.any(r_check > max_r_bound + 1e-9):
                scale *= 0.9
                continue
            
            # Check overlaps
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r_check[i_idx] + r_check[j_idx]
            if np.any(sum_r > dist + 1e-9):
                scale *= 0.9
                continue
            
            # Valid
            break
            
        r = r * scale
        best_state[:, 2] = r
    
    return best_state

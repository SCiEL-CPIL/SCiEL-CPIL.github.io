import numpy as np
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit

def construct():
    start_time = time.time()
    
    def check_validity(x, y, r, tol=1e-8):
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        if np.any(r < -tol):
            return False
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        
        if np.any(dist < min_dist - tol):
            return False
        return True

    def calculate_score(x, y, r):
        return np.sum(r)

    def expand_and_push(x, y, r, iterations=500):
        """
        Iteratively expand radii to maximum possible and push overlapping circles apart.
        """
        for _ in range(iterations):
            if time.time() > DEADLINE:
                break
            
            # 1. Expand radii
            new_r = np.zeros(N)
            for i in range(N):
                # Wall limits
                max_r_wall = min(x[i], 1 - x[i], y[i], 1 - y[i])
                if max_r_wall < 0:
                    max_r_wall = 0
                    
                # Neighbor limits
                dx = x[i] - x
                dy = y[i] - y
                dists = np.sqrt(dx**2 + dy**2)
                dists[i] = np.inf
                neighbor_limits = dists - r
                max_r_neighbor = np.min(neighbor_limits)
                
                new_r[i] = max(0.0, min(max_r_wall, max_r_neighbor))
            
            r = new_r
            
            # 2. Push overlapping circles apart
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            
            np.fill_diagonal(dist, np.inf)
            
            # Inner push iterations
            for _ in range(10):
                overlaps = dist < min_dist
                if not np.any(overlaps):
                    break
                
                i_idx, j_idx = np.where(overlaps)
                
                # To ensure we don't miss overlaps due to stale matrix, 
                # we process all current overlaps.
                # Note: This is a heuristic.
                for k in range(len(i_idx)):
                    i = i_idx[k]
                    j = j_idx[k]
                    
                    dx_ij = x[i] - x[j]
                    dy_ij = y[i] - y[j]
                    d = np.sqrt(dx_ij**2 + dy_ij**2)
                    
                    if d < 1e-10:
                        x[i] += 0.01
                        y[i] += 0.01
                        continue
                    
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        push_x = (dx_ij / d) * (overlap / 2)
                        push_y = (dy_ij / d) * (overlap / 2)
                        
                        x[i] += push_x
                        y[i] += push_y
                        x[j] -= push_x
                        y[j] -= push_y
                        
                        x[i] = np.clip(x[i], r[i], 1 - r[i])
                        y[i] = np.clip(y[i], r[i], 1 - r[i])
                        x[j] = np.clip(x[j], r[j], 1 - r[j])
                        y[j] = np.clip(y[j], r[j], 1 - r[j])
                
                # Recalculate dist for next inner iteration
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[:, None] + r[None, :]
                np.fill_diagonal(dist, np.inf)

        return x, y, r

    best_x = None
    best_y = None
    best_r = None
    best_val = 0.0

    # Start 1: Grid (5x5 + 1)
    side = 5
    grid_x = []
    grid_y = []
    grid_r = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            grid_x.append(x)
            grid_y.append(y)
            grid_r.append(r)
            idx += 1
    
    x0 = np.array(grid_x)
    y0 = np.array(grid_y)
    r0 = np.array(grid_r)
    
    # Add 26th circle
    x0 = np.append(x0, 0.5)
    y0 = np.append(y0, 0.5)
    r0 = np.append(r0, 0.01)
    
    x_opt, y_opt, r_opt = expand_and_push(x0.copy(), y0.copy(), r0.copy(), iterations=1000)
    if check_validity(x_opt, y_opt, r_opt):
        val = calculate_score(x_opt, y_opt, r_opt)
        if val > best_val:
            best_val = val
            best_x, best_y, best_r = x_opt, y_opt, r_opt

    # Random Restarts
    rng = np.random.RandomState(42)
    
    while time.time() < DEADLINE - 5:
        x_rand = rng.uniform(0.1, 0.9, N)
        y_rand = rng.uniform(0.1, 0.9, N)
        r_rand = rng.uniform(0.01, 0.03, N)
        
        x_opt, y_opt, r_opt = expand_and_push(x_rand.copy(), y_rand.copy(), r_rand.copy(), iterations=500)
        
        if check_validity(x_opt, y_opt, r_opt):
            val = calculate_score(x_opt, y_opt, r_opt)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x_opt, y_opt, r_opt

    if best_x is None:
        best_x = x0
        best_y = y0
        best_r = r0
        best_x, best_y, best_r = expand_and_push(best_x.copy(), best_y.copy(), best_r.copy(), iterations=100)

    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 10):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = best_r * 0.8

    return np.column_stack((best_x, best_y, best_r))

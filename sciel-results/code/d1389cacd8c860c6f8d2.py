import numpy as np
import random
import math

N = 21

def construct():
    best_score = -1
    best_result = None
    
    # Try a few aspect ratios
    # Perimeter 4 => W + H = 2
    # Square W=1, H=1 is a strong candidate.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for w_ratio in aspect_ratios:
        W = w_ratio
        H = 2.0 - W
        
        # Initialization: Grid
        # 5x5 grid = 25 spots. We need 21.
        # Let's use a 5x4 grid (20) + 1 center?
        # Or just 21 random points?
        # Grid is better for start.
        # 5 cols, 5 rows.
        cols = 5
        rows = 5
        x_init = []
        y_init = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x_init.append((i + 0.5) * W / cols)
                y_init.append((j + 0.5) * H / rows)
                count += 1
        
        x = np.array(x_init)
        y = np.array(y_init)
        
        # Simulated Annealing
        def get_radii(x, y, W, H):
            r_wall = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)
            np.fill_diagonal(dist, np.inf)
            min_dist = np.min(dist, axis=1)
            r_neighbor = min_dist / 2.0
            return np.minimum(r_wall, r_neighbor)
        
        current_score = np.sum(get_radii(x, y, W, H))
        
        # SA Parameters
        T_start = 0.1
        T_end = 0.0001
        iterations = 50000
        
        for _ in range(3): # 3 restarts per aspect ratio
            x_curr = x.copy()
            y_curr = y.copy()
            # Add small random noise to break symmetry
            x_curr += np.random.uniform(-0.01, 0.01, N)
            y_curr += np.random.uniform(-0.01, 0.01, N)
            x_curr = np.clip(x_curr, 0, W)
            y_curr = np.clip(y_curr, 0, H)
            
            r_curr = get_radii(x_curr, y_curr, W, H)
            score_curr = np.sum(r_curr)
            
            best_local_score = score_curr
            best_local_x = x_curr.copy()
            best_local_y = y_curr.copy()
            
            for it in range(iterations):
                # Cooling
                T = T_start * (T_end / T_start) ** (it / iterations)
                
                # Pick random circle
                k = np.random.randint(N)
                
                # Propose move
                # Step size should be adaptive or fixed
                step = 0.05 * (T / T_start + 0.1) # Roughly
                # Or just fixed step with SA acceptance
                step = 0.02
                
                old_xk = x_curr[k]
                old_yk = y_curr[k]
                
                x_curr[k] += np.random.normal(0, step)
                y_curr[k] += np.random.normal(0, step)
                
                # Clamp
                x_curr[k] = np.clip(x_curr[k], 0, W)
                y_curr[k] = np.clip(y_curr[k], 0, H)
                
                r_new = get_radii(x_curr, y_curr, W, H)
                score_new = np.sum(r_new)
                
                delta = score_new - score_curr
                
                if delta > 0 or np.random.random() < math.exp(delta / T):
                    score_curr = score_new
                    r_curr = r_new
                    if score_new > best_local_score:
                        best_local_score = score_new
                        best_local_x = x_curr.copy()
                        best_local_y = y_curr.copy()
                else:
                    x_curr[k] = old_xk
                    y_curr[k] = old_yk
            
            if best_local_score > best_score:
                best_score = best_local_score
                best_result = np.column_stack((best_local_x, best_local_y, get_radii(best_local_x, best_local_y, W, H)))
                
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result

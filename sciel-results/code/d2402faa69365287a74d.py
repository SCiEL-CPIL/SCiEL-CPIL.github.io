import numpy as np
import random

N = 21

def construct():
    # Initial grid in 1x1 square
    side = int(np.ceil(np.sqrt(N)))
    # 5x5 grid
    cols = 5
    rows = 5
    idx = 0
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(cols):
        for j in range(rows):
            if idx >= N:
                break
            x[idx] = (i + 0.5) / cols
            y[idx] = (j + 0.5) / rows
            r[idx] = 0.1
            idx += 1
            
    W = 1.0
    H = 1.0
    
    def energy_vec(x, y, r, W, H):
        val = -np.sum(r)
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        rad_sum = r[:, None] + r[None, :]
        
        # Overlap penalty
        overlap_pen = np.sum(np.maximum(0, rad_sum**2 - dist_sq))
        
        # Bounds
        bound_pen = 0.0
        bound_pen += np.sum(np.maximum(0, r - x)**2)
        bound_pen += np.sum(np.maximum(0, x - (W - r))**2)
        bound_pen += np.sum(np.maximum(0, r - y)**2)
        bound_pen += np.sum(np.maximum(0, y - (H - r))**2)
        
        return val + 100.0 * (overlap_pen + bound_pen)

    # SA Parameters
    T = 0.1
    T_min = 1e-6
    alpha = 0.99
    current_x = x.copy()
    current_y = y.copy()
    current_r = r.copy()
    current_E = energy_vec(current_x, current_y, current_r, W, H)
    
    best_x = current_x.copy()
    best_y = current_y.copy()
    best_r = current_r.copy()
    best_E = current_E
    
    max_iter = 20000
    step_size = 0.05
    
    for k in range(max_iter):
        # Propose move
        i = random.randint(0, N-1)
        move_type = random.choice(['x', 'y', 'r'])
        
        new_x = current_x.copy()
        new_y = current_y.copy()
        new_r = current_r.copy()
        
        if move_type == 'x':
            new_x[i] += random.uniform(-step_size, step_size)
        elif move_type == 'y':
            new_y[i] += random.uniform(-step_size, step_size)
        else:
            # Increase radius slightly
            new_r[i] += random.uniform(-0.01, 0.02)
            if new_r[i] < 0.01: new_r[i] = 0.01
            
        new_E = energy_vec(new_x, new_y, new_r, W, H)
        
        # Acceptance criterion
        if new_E < current_E or random.random() < np.exp((current_E - new_E) / T):
            current_x = new_x
            current_y = new_y
            current_r = new_r
            current_E = new_E
            
            if current_E < best_E:
                best_x = current_x.copy()
                best_y = current_y.copy()
                best_r = current_r.copy()
                best_E = current_E
                
        T *= alpha
        
        # Reduce step size over time
        step_size *= 0.9995
        
        if T < T_min:
            break
            
    # Final validation and cleanup
    def is_valid(x, y, r, W, H):
        for i in range(N):
            if r[i] <= 0: return False
            if x[i] - r[i] < -1e-8 or x[i] + r[i] > W + 1e-8: return False
            if y[i] - r[i] < -1e-8 or y[i] + r[i] > H + 1e-8: return False
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-8:
                    return False
        return True

    if not is_valid(best_x, best_y, best_r, W, H):
        # Try to repair by shrinking radii
        for i in range(N):
            if best_x[i] - best_r[i] < 0:
                best_r[i] = best_x[i]
            if best_x[i] + best_r[i] > W:
                best_r[i] = W - best_x[i]
            if best_y[i] - best_r[i] < 0:
                best_r[i] = min(best_r[i], best_y[i])
            if best_y[i] + best_r[i] > H:
                best_r[i] = min(best_r[i], H - best_y[i])
                
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((best_x[i]-best_x[j])**2 + (best_y[i]-best_y[j])**2)
                rad_sum = best_r[i] + best_r[j]
                if dist < rad_sum:
                    shrink = (rad_sum - dist) / 2.0 + 1e-8
                    best_r[i] -= shrink
                    best_r[j] -= shrink
                    if best_r[i] < 0.01: best_r[i] = 0.01
                    if best_r[j] < 0.01: best_r[j] = 0.01
                    
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result

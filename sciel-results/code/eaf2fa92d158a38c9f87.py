import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Initial seed result (grid) as fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    best_result = np.array(cells[:N])
    best_score = np.sum(best_result[:, 2])

    def vectorized_objective(z):
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        
        obj = -np.sum(r)
        
        # Overlap
        dx = np.subtract.outer(x, x)
        dy = np.subtract.outer(y, y)
        dist = np.sqrt(dx**2 + dy**2)
        
        sum_r = np.add.outer(r, r)
        
        overlap = sum_r - dist
        overlap_pos = np.where(overlap > 0, overlap, 0)
        
        iu, ju = np.triu_indices(N, k=1)
        penalty_overlap = np.sum(overlap_pos[iu, ju]**2)
        
        # Boundary
        px_low = np.maximum(0, r - x)
        px_high = np.maximum(0, x - (1 - r))
        py_low = np.maximum(0, r - y)
        py_high = np.maximum(0, y - (1 - r))
        
        penalty_boundary = np.sum(px_low**2) + np.sum(px_high**2) + np.sum(py_low**2) + np.sum(py_high**2)
        
        return obj + 1000.0 * (penalty_overlap + penalty_boundary)

    def is_valid(x, y, r):
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
            return False
        if np.any(r < 0):
            return False
            
        dx = np.subtract.outer(x, x)
        dy = np.subtract.outer(y, y)
        dist = np.sqrt(dx**2 + dy**2)
        sum_r = np.add.outer(r, r)
        
        # Check if any overlap > 0
        # Use a small epsilon for floating point
        if np.any(sum_r - dist > 1e-9):
             # Only check i < j to be consistent, but sum_r - dist is symmetric
             # We can just check the upper triangle
             iu, ju = np.triu_indices(N, k=1)
             if np.any(sum_r[iu, ju] - dist[iu, ju] > 1e-9):
                 return False
        return True

    def repair(x, y, r):
        # Clamp positions to valid box for current radii
        # If r > 0.5, invalid
        if np.any(r > 0.5):
            return None
            
        x_new = np.clip(x, r, 1 - r)
        y_new = np.clip(y, r, 1 - r)
        
        if is_valid(x_new, y_new, r):
            return x_new, y_new, r
            
        # Shrink radii to resolve overlaps
        r_curr = r.copy()
        x_curr = x_new.copy()
        y_curr = y_new.copy()
        
        for _ in range(20):
            if is_valid(x_curr, y_curr, r_curr):
                return x_curr, y_curr, r_curr
                
            # Find max overlap
            dx = np.subtract.outer(x_curr, x_curr)
            dy = np.subtract.outer(y_curr, y_curr)
            dist = np.sqrt(dx**2 + dy**2)
            sum_r = np.add.outer(r_curr, r_curr)
            
            iu, ju = np.triu_indices(N, k=1)
            overlaps = sum_r[iu, ju] - dist[iu, ju]
            
            max_overlap_idx = np.argmax(overlaps)
            if overlaps[max_overlap_idx] <= 0:
                return x_curr, y_curr, r_curr # Should be valid then?
                
            i = iu[max_overlap_idx]
            j = ju[max_overlap_idx]
            
            # Shrink both
            # d = dist[i,j], r_sum = r[i]+r[j]
            # We need r[i]+r[j] <= d
            # Scale factor
            d = dist[i, j]
            if d < 1e-9:
                # Centers same, shrink one to 0?
                r_curr[i] = 0
                continue
                
            factor = (d / (r_curr[i] + r_curr[j])) * 0.99
            r_curr[i] *= factor
            r_curr[j] *= factor
            
            # Update positions? No, positions are fixed in this repair step.
            # If we don't move them, shrinking radii is sufficient.
            
        return x_curr, y_curr, r_curr

    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
    for i in range(N):
        bounds.append((0, 1)) # y
    for i in range(N):
        bounds.append((0, 0.5)) # r

    # Restart loop
    while time.time() < deadline:
        # Random initialization
        # Use a mix of grid and random
        if np.random.rand() < 0.5:
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
        else:
            # Grid with jitter
            side = int(np.ceil(np.sqrt(N)))
            idx = np.arange(N)
            x = (idx % side + 0.5) / side + np.random.uniform(-0.05, 0.05, N)
            y = (idx // side + 0.5) / side + np.random.uniform(-0.05, 0.05, N)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
        r = np.random.uniform(0.02, 0.05, N)
        
        z0 = np.concatenate([x, y, r])
        
        res = minimize(vectorized_objective, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10})
        
        z_opt = res.x
        x_opt = z_opt[0:N]
        y_opt = z_opt[N:2*N]
        r_opt = z_opt[2*N:3*N]
        
        res_repaired = repair(x_opt, y_opt, r_opt)
        
        if res_repaired:
            x_f, y_f, r_f = res_repaired
            score = np.sum(r_f)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x_f, y_f, r_f))
        
        if is_valid(x_opt, y_opt, r_opt):
            score = np.sum(r_opt)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x_opt, y_opt, r_opt))

    return best_result

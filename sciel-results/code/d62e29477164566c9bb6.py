import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns base positions for 64 tubes.
    """
    N = 8
    start_time = time.time()
    time_limit = 600.0
    margin = 60.0
    
    # Helper: Compute union area of M axis-aligned rectangles
    # rects: array of shape (M, 4) -> [x0, x1, y0, y1]
    def union_area(rects):
        if len(rects) == 0:
            return 0.0
        
        x0s = rects[:, 0]
        x1s = rects[:, 1]
        y0s = rects[:, 2]
        y1s = rects[:, 3]
        
        x_coords = np.unique(np.concatenate([x0s, x1s]))
        
        area = 0.0
        for i in range(len(x_coords) - 1):
            x_left = x_coords[i]
            x_right = x_coords[i + 1]
            width = x_right - x_left
            
            mask = (x0s <= x_left) & (x1s >= x_right)
            
            if not np.any(mask):
                continue
                
            active_y0s = y0s[mask]
            active_y1s = y1s[mask]
            
            idx = np.argsort(active_y0s)
            active_y0s = active_y0s[idx]
            active_y1s = active_y1s[idx]
            
            y_len = 0.0
            curr_start = active_y0s[0]
            curr_end = active_y1s[0]
            
            for j in range(1, len(active_y0s)):
                if active_y0s[j] <= curr_end:
                    if active_y1s[j] > curr_end:
                        curr_end = active_y1s[j]
                else:
                    y_len += curr_end - curr_start
                    curr_start = active_y0s[j]
                    curr_end = active_y1s[j]
            y_len += curr_end - curr_start
            
            area += width * y_len
            
        return area

    # Objective function
    c_grid = np.linspace(0, 1, 50)
    dc = c_grid[1] - c_grid[0]
    
    def objective(p):
        x = p[:N]
        y = p[N:]
        
        total_vol = 0.0
        
        for c in c_grid:
            rects = np.empty((N*N, 4))
            k = 0
            for i in range(N):
                rx0 = x[i] + c * i / N
                rx1 = rx0 + 1.0 / N
                for j in range(N):
                    ry0 = y[j] + c * j / N
                    ry1 = ry0 + 1.0 / N
                    rects[k] = [rx0, rx1, ry0, ry1]
                    k += 1
            
            total_vol += union_area(rects) * dc
        
        return total_vol

    # Initial guess: Keich construction for x and y
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x0 = keich.copy()
    y0 = keich.copy()
    p0 = np.concatenate([x0, y0])
    
    best_p = p0.copy()
    best_obj = objective(p0)
    
    # Optimization loop with restarts
    rng = np.random.default_rng(42)
    
    initial_points = [p0.copy()]
    
    for _ in range(15):
        perturbed = p0 + rng.uniform(-0.3, 0.3, size=2*N)
        initial_points.append(perturbed)
        
    initial_points.append(np.linspace(0, 1, 2*N))
    initial_points.append(rng.uniform(0, 1, size=2*N))
    
    for p_init in initial_points:
        if time.time() - start_time > time_limit - margin:
            break
            
        try:
            res = opt.minimize(objective, p_init, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
            
            obj_val = objective(res.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_p = res.x.copy()
        except Exception:
            continue
            
    # Final polish
    if time.time() - start_time < time_limit - margin:
        try:
            res_polish = opt.minimize(objective, best_p, method='Nelder-Mead',
                                      options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9})
            obj_val = objective(res_polish.x)
            if obj_val < best_obj:
                best_p = res_polish.x
        except Exception:
            pass
            
    x_opt = best_p[:N]
    y_opt = best_p[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[i_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

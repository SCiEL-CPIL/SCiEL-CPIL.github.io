import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 115.0

    best_result = None
    best_score = -np.inf
    best_params = None # (W, rows, cols)

    # Coarse search
    W_candidates = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    grid_configs = [(4, 6), (5, 5), (3, 7), (6, 4), (7, 3)]
    
    for W in W_candidates:
        if time.time() > deadline - 20:
            break
        H = 2.0 - W
        
        for rows, cols in grid_configs:
            if time.time() > deadline - 20:
                break
            
            dx = W / cols
            dy = H / rows
            
            xs = []
            ys = []
            for i in range(rows):
                for j in range(cols):
                    if len(xs) >= N:
                        break
                    x = (j + 0.5) * dx
                    y = (i + 0.5) * dy
                    xs.append(x)
                    ys.append(y)
            
            if len(xs) < N:
                continue
                
            xs = np.array(xs[:N])
            ys = np.array(ys[:N])
            r_init = np.min([dx, dy]) * 0.48
            rs = np.full(N, r_init)
            
            v0 = np.concatenate([xs, ys, rs])
            
            bounds = []
            for i in range(N): bounds.append((0, W))
            for i in range(N): bounds.append((0, H))
            for i in range(N): bounds.append((0, 1.0))
                
            def objective(v):
                x = v[:N]
                y = v[N:2*N]
                r = v[2*N:3*N]
                
                pen_wall = np.sum(np.maximum(0, r - x)**2)
                pen_wall += np.sum(np.maximum(0, r - (W - x))**2)
                pen_wall += np.sum(np.maximum(0, r - y)**2)
                pen_wall += np.sum(np.maximum(0, r - (H - y))**2)
                
                diff_x = x[:, None] - x[None, :]
                diff_y = y[:, None] - y[None, :]
                dist = np.sqrt(diff_x**2 + diff_y**2)
                sum_r = r[:, None] + r[None, :]
                violations = np.maximum(0, sum_r - dist)
                pen_overlap = np.sum(violations**2)
                
                return -np.sum(r) + 100.0 * (pen_wall + pen_overlap)
                
            try:
                res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 150, 'ftol': 1e-8})
                
                if res.fun < best_score + 1e-6:
                    x_opt = res.x[:N]
                    y_opt = res.x[N:2*N]
                    r_opt = res.x[2*N:3*N]
                    
                    valid = True
                    if np.any(r_opt <= 0) or np.any(x_opt < r_opt) or np.any(x_opt > W - r_opt) or np.any(y_opt < r_opt) or np.any(y_opt > H - r_opt):
                        valid = False
                    else:
                        diff_x = x_opt[:, None] - x_opt[None, :]
                        diff_y = y_opt[:, None] - y_opt[None, :]
                        dist = np.sqrt(diff_x**2 + diff_y**2)
                        sum_r = r_opt[:, None] + r_opt[None, :]
                        if np.any(sum_r > dist + 1e-5):
                            valid = False
                            
                    if valid:
                        score = np.sum(r_opt)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack((x_opt, y_opt, r_opt))
                            best_params = (W, rows, cols)
                            
            except Exception:
                pass

    # Polish step
    if best_params is not None and time.time() < deadline - 10:
        W, rows, cols = best_params
        H = 2.0 - W
        # Restart from best_result
        v0 = best_result.flatten()
        
        bounds = []
        for i in range(N): bounds.append((0, W))
        for i in range(N): bounds.append((0, H))
        for i in range(N): bounds.append((0, 1.0))
        
        def objective_polish(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            
            pen_wall = np.sum(np.maximum(0, r - x)**2)
            pen_wall += np.sum(np.maximum(0, r - (W - x))**2)
            pen_wall += np.sum(np.maximum(0, r - y)**2)
            pen_wall += np.sum(np.maximum(0, r - (H - y))**2)
            
            diff_x = x[:, None] - x[None, :]
            diff_y = y[:, None] - y[None, :]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            sum_r = r[:, None] + r[None, :]
            violations = np.maximum(0, sum_r - dist)
            pen_overlap = np.sum(violations**2)
            
            return -np.sum(r) + 1000.0 * (pen_wall + pen_overlap)
            
        try:
            res = minimize(objective_polish, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.fun < best_score + 1e-6:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                valid = True
                if np.any(r_opt <= 0) or np.any(x_opt < r_opt) or np.any(x_opt > W - r_opt) or np.any(y_opt < r_opt) or np.any(y_opt > H - r_opt):
                    valid = False
                else:
                    diff_x = x_opt[:, None] - x_opt[None, :]
                    diff_y = y_opt[:, None] - y_opt[None, :]
                    dist = np.sqrt(diff_x**2 + diff_y**2)
                    sum_r = r_opt[:, None] + r_opt[None, :]
                    if np.any(sum_r > dist + 1e-5):
                        valid = False
                        
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_opt, y_opt, r_opt))
        except Exception:
            pass

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        
    return best_result

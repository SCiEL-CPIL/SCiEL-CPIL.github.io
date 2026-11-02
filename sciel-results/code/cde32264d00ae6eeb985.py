import numpy as np
import time
import scipy.optimize

N = 21
DEADLINE = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    best_val = -np.inf
    best_sol = None
    
    # Try several restarts with different initial aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.66, 1.1, 0.9]
    
    for aspect in aspects:
        if time.time() - start_time > DEADLINE:
            break
            
        # Initial W, H
        # W + H <= 2
        # W = aspect * H
        # H(aspect + 1) <= 2 => H <= 2 / (aspect + 1)
        # Start slightly smaller
        H0 = 1.9 / (aspect + 1)
        W0 = aspect * H0
        
        # Initialize grid
        # 5x5 grid
        rows, cols = 5, 5
        x_coords = np.linspace(0.1, W0 - 0.1, cols)
        y_coords = np.linspace(0.1, H0 - 0.1, rows)
        
        init_x = []
        init_y = []
        init_r = []
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= N:
                    break
                init_x.append(x_coords[j])
                init_y.append(y_coords[i])
                cell_w = W0 / cols
                cell_h = H0 / rows
                r0 = 0.4 * min(cell_w, cell_h)
                init_r.append(r0)
                count += 1
        
        x0 = np.array(init_x)
        y0 = np.array(init_y)
        r0 = np.array(init_r)
        
        v0 = np.concatenate([x0, y0, r0, [W0, H0]])
        
        bounds = []
        for _ in range(N):
            bounds.append((1e-6, 2.0)) # x
            bounds.append((1e-6, 2.0)) # y
            bounds.append((1e-6, 1.0)) # r
        bounds.append((0.1, 2.0)) # W
        bounds.append((0.1, 2.0)) # H
        
        def objective(v):
            x = v[0:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            W = v[3*N]
            H = v[3*N + 1]
            
            val = -np.sum(r)
            
            # Overlap penalty
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            r_sum = r[:, None] + r[None, :]
            
            iu = np.triu_indices(N, k=1)
            overlaps = np.maximum(0, r_sum[iu] - dist[iu])
            overlap_penalty = np.sum(overlaps**2)
            
            # Boundary penalty
            bx = np.maximum(0, -x) + np.maximum(0, x - W)
            by = np.maximum(0, -y) + np.maximum(0, y - H)
            bound_penalty = np.sum(bx**2) + np.sum(by**2)
            
            # Perimeter penalty
            perim_viol = np.maximum(0, 2*(W + H) - 4)
            perim_penalty = perim_viol**2
            
            w_overlap = 1000.0
            w_bound = 1000.0
            w_perim = 1000.0
            
            return val + w_overlap * overlap_penalty + w_bound * bound_penalty + w_perim * perim_penalty

        try:
            res = scipy.optimize.minimize(objective, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            v_opt = res.x
        except Exception:
            continue
            
        x_opt = v_opt[0:N]
        y_opt = v_opt[N:2*N]
        r_opt = v_opt[2*N:3*N]
        W_opt = v_opt[3*N]
        H_opt = v_opt[3*N + 1]
        
        # Enforce Perimeter
        if 2 * (W_opt + H_opt) > 4.0:
            scale = 4.0 / (2 * (W_opt + H_opt))
            W_opt *= scale
            H_opt *= scale
            x_opt *= scale
            y_opt *= scale
            r_opt *= scale
            
        # Check Validity and Repair
        valid = True
        
        # Check Bounds
        if np.any(x_opt < -1e-8) or np.any(x_opt > W_opt + 1e-8) or \
           np.any(y_opt < -1e-8) or np.any(y_opt > H_opt + 1e-8):
            x_opt = np.clip(x_opt, 0, W_opt)
            y_opt = np.clip(y_opt, 0, H_opt)
            valid = False # Need to re-check overlaps after clipping
            
        # Check Overlaps
        dx = x_opt[:, None] - x_opt[None, :]
        dy = y_opt[:, None] - y_opt[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r_opt[:, None] + r_opt[None, :]
        iu = np.triu_indices(N, k=1)
        overlaps = r_sum[iu] - dist[iu]
        
        if np.any(overlaps > 1e-8):
            valid = False
            # Repair by shrinking radii
            # We can also try to move centers, but that's complex.
            # Shrinking is safe.
            for _ in range(50):
                if time.time() - start_time > DEADLINE:
                    break
                r_opt *= 0.95
                r_sum = r_opt[:, None] + r_opt[None, :]
                overlaps = r_sum[iu] - dist[iu]
                if not np.any(overlaps > 1e-8):
                    valid = True
                    break
                    # Note: dist is fixed here.
        
        if valid:
            val = np.sum(r_opt)
            if val > best_val:
                best_val = val
                best_sol = np.column_stack([x_opt, y_opt, r_opt])

    if best_sol is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_sol

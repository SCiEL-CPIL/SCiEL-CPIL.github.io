import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    best_val = -np.inf
    best_result = None
    
    # Helper to check feasibility and calculate value
    def evaluate(x):
        """
        x: [w, r_1..r_N, cx_1..cx_N, cy_1..cy_N]
        Returns (value, is_feasible)
        """
        w = x[0]
        h = 2.0 - w
        r = x[1:1+N]
        cx = x[1+N:1+2*N]
        cy = x[1+2*N:1+3*N]
        
        # Check basic bounds
        if w <= 0.1 or h <= 0.1:
            return 0.0, False
        if np.any(r < -1e-6):
            return 0.0, False
        r = np.maximum(r, 0)
        
        # Check boundaries
        if np.any(cx - r < -1e-6) or np.any(w - cx - r < -1e-6):
            return 0.0, False
        if np.any(cy - r < -1e-6) or np.any(h - cy - r < -1e-6):
            return 0.0, False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (cx[i]-cx[j])**2 + (cy[i]-cy[j])**2
                min_dist_sq = (r[i]+r[j])**2
                if dist_sq < min_dist_sq - 1e-6:
                    return 0.0, False
        
        return np.sum(r), True

    # LP Polish: Given fixed w and centers, maximize sum of radii
    def polish(w, cx, cy):
        """
        Solves LP to find optimal radii for fixed centers and width.
        Returns optimal radii array.
        """
        h = 2.0 - w
        
        # Objective: minimize -sum(r)
        c_obj = -np.ones(N)
        
        # Constraints A_ub r <= b_ub
        A_rows = []
        b_vals = []
        
        # 1. Boundary constraints: r_i <= dist_to_wall_i
        for i in range(N):
            dist_wall = min(cx[i], w - cx[i], cy[i], h - cy[i])
            if dist_wall < 0:
                # Infeasible center, return zeros or handle gracefully
                return np.zeros(N)
            A_row = np.zeros(N)
            A_row[i] = 1.0
            A_rows.append(A_row)
            b_vals.append(dist_wall)
            
        # 2. Pairwise constraints: r_i + r_j <= dist_ij
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                A_row = np.zeros(N)
                A_row[i] = 1.0
                A_row[j] = 1.0
                A_rows.append(A_row)
                b_vals.append(dist)
                
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        
        bounds = [(0, None) for _ in range(N)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.x
            else:
                return np.zeros(N)
        except Exception:
            return np.zeros(N)

    def objective(x):
        w = x[0]
        r = x[1:1+N]
        return -np.sum(r)

    def constraints(x):
        w = x[0]
        h = 2.0 - w
        r = x[1:1+N]
        cx = x[1+N:1+2*N]
        cy = x[1+2*N:1+3*N]
        
        cons = []
        
        # w bounds
        cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.5}) # w >= 0.5
        cons.append({'type': 'ineq', 'fun': lambda x: 1.5 - x[0]}) # w <= 1.5
        
        # r bounds (positive)
        for i in range(N):
            idx = 1 + i
            cons.append({'type': 'ineq', 'fun': lambda x, idx=idx: x[idx]}) # r_i >= 0
            
        # Boundary constraints
        for i in range(N):
            r_idx = 1 + i
            cx_idx = 1 + N + i
            cy_idx = 1 + 2*N + i
            
            # Left
            cons.append({'type': 'ineq', 'fun': lambda x, cx=cx_idx, r=r_idx: x[cx] - x[r]})
            # Right
            cons.append({'type': 'ineq', 'fun': lambda x, w=0, cx=cx_idx, r=r_idx: x[w] - x[cx] - x[r]})
            # Bottom
            cons.append({'type': 'ineq', 'fun': lambda x, cy=cy_idx, r=r_idx: x[cy] - x[r]})
            # Top
            cons.append({'type': 'ineq', 'fun': lambda x, w=0, cy=cy_idx, r=r_idx: (2.0 - x[w]) - x[cy] - x[r]})
            
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                ri_idx = 1 + i
                rj_idx = 1 + j
                cxi_idx = 1 + N + i
                cxj_idx = 1 + N + j
                cyi_idx = 1 + 2*N + i
                cyj_idx = 1 + 2*N + j
                
                def non_overlap(x, ri=ri_idx, rj=rj_idx, cxi=cxi_idx, cxj=cxj_idx, cyi=cyi_idx, cyj=cyj_idx):
                    dx = x[cxi] - x[cxj]
                    dy = x[cyi] - x[cyj]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (x[ri] + x[rj])**2
                    return dist_sq - min_dist_sq
                
                cons.append({'type': 'ineq', 'fun': non_overlap})
                
        return cons

    def get_bounds():
        bounds = [(0.5, 1.5)] # w
        for _ in range(N):
            bounds.append((0.0, 1.0)) # r
        for _ in range(N):
            bounds.append((0.0, 2.0)) # cx
        for _ in range(N):
            bounds.append((0.0, 2.0)) # cy
        return bounds

    # Initializers
    def init_grid(w, rows, cols):
        h = 2.0 - w
        x = np.zeros(3*N + 1)
        x[0] = w
        cw = w / cols
        ch = h / rows
        r_init = 0.45 * min(cw, ch)
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= N:
                    break
                cx = (j + 0.5) * cw
                cy = (i + 0.5) * ch
                r = min(r_init, cx, w-cx, cy, h-cy) * 0.99
                r = max(r, 1e-4)
                
                x[1+count] = r
                x[1+N+count] = cx
                x[1+2*N+count] = cy
                count += 1
                
        # If not filled, place rest randomly or at center
        while count < N:
            cx = w/2 + np.random.uniform(-w/4, w/4)
            cy = h/2 + np.random.uniform(-h/4, h/4)
            r = 1e-4
            x[1+count] = r
            x[1+N+count] = cx
            x[1+2*N+count] = cy
            count += 1
        return x

    def init_random(w):
        h = 2.0 - w
        x = np.zeros(3*N + 1)
        x[0] = w
        for i in range(N):
            cx = np.random.uniform(0.1*w, 0.9*w)
            cy = np.random.uniform(0.1*h, 0.9*h)
            r = 1e-4
            x[1+i] = r
            x[1+N+i] = cx
            x[1+2*N+i] = cy
        return x

    configs = [
        (1.0, 5, 5),
        (1.0, 4, 6),
        (1.2, 4, 6),
        (0.8, 6, 4),
        (1.0, 3, 7),
        (1.0, 7, 3),
        (1.1, 5, 5),
        (0.9, 5, 5),
        (1.3, 4, 6),
        (0.7, 6, 4),
        (1.0, 2, 11),
        (1.0, 11, 2),
        (1.4, 3, 8),
        (0.6, 8, 3),
    ]
    
    # Add some random starts
    for _ in range(5):
        configs.append((np.random.uniform(0.7, 1.3), None, None))

    bounds = get_bounds()
    cons = constraints
    
    for w_init, rows, cols in configs:
        if time.time() > deadline - 5:
            break
            
        if rows is None:
            x0 = init_random(w_init)
        else:
            x0 = init_grid(w_init, rows, cols)
            
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            x_opt = res.x
            
            # Verify SLSQP result
            val, is_feas = evaluate(x_opt)
            
            if is_feas:
                # Apply LP Polish
                w_opt = x_opt[0]
                cx_opt = x_opt[1+N:1+2*N]
                cy_opt = x_opt[1+2*N:1+3*N]
                
                r_polished = polish(w_opt, cx_opt, cy_opt)
                
                # Construct final x with polished radii
                x_final = x_opt.copy()
                x_final[1:1+N] = r_polished
                
                val_final, is_feas_final = evaluate(x_final)
                
                if is_feas_final and val_final > best_val:
                    best_val = val_final
                    best_result = x_final
                    
        except Exception:
            pass

    if best_result is None:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (j + 0.5) * (w/side)
                cy = (i + 0.5) * (h/side)
                r = 0.5/side - 1e-3
                cells.append([cx, cy, r])
                count += 1
        return np.array(cells)
        
    # Format output: array of shape (n, 3) [x, y, r]
    w = best_result[0]
    r = best_result[1:1+N]
    cx = best_result[1+N:1+2*N]
    cy = best_result[1+2*N:1+3*N]
    
    result = np.column_stack((cx, cy, r))
    return result

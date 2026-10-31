import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    # Helper to create a valid fallback solution
    # Grid of 5x5, pick 21 positions, small radii
    def get_fallback_solution(w=1.0):
        h = 2.0 - w
        # 5x5 grid
        cols = 5
        rows = 5
        xs = np.linspace(w / (2*cols), w - w / (2*cols), cols)
        ys = np.linspace(h / (2*rows), h - h / (2*rows), rows)
        
        # Create meshgrid
        X, Y = np.meshgrid(xs, ys)
        X = X.flatten()
        Y = Y.flatten()
        
        # Take first 21
        X = X[:N]
        Y = Y[:N]
        
        # Max possible radius for this grid spacing
        dx = w / cols
        dy = h / rows
        r_max = min(dx, dy) * 0.49
        # Use a slightly smaller radius to ensure strict non-overlap and boundary respect
        r = np.full(N, r_max * 0.9)
        
        sol = np.column_stack((X, Y, r))
        return sol

    # Vectorized constraint function
    # v: [w, x1, y1, r1, ..., xN, yN, rN]
    def constraints(v):
        w = v[0]
        h = 2.0 - w
        
        # Extract coordinates and radii
        # Indices: 1, 2, 3 for circle 0; 4, 5, 6 for circle 1, etc.
        # x: 1, 4, 7... -> 1 + 3*i
        # y: 2, 5, 8... -> 2 + 3*i
        # r: 3, 6, 9... -> 3 + 3*i
        
        idx_x = np.arange(1, 3*N + 1, 3)
        idx_y = np.arange(2, 3*N + 1, 3)
        idx_r = np.arange(3, 3*N + 1, 3)
        
        xs = v[idx_x]
        ys = v[idx_y]
        rs = v[idx_r]
        
        cons = []
        
        # Boundary constraints:
        # x_i >= r_i  => x_i - r_i >= 0
        cons.extend(xs - rs)
        # w - x_i >= r_i => w - x_i - r_i >= 0
        cons.extend(w - xs - rs)
        # y_i >= r_i => y_i - r_i >= 0
        cons.extend(ys - rs)
        # h - y_i >= r_i => h - y_i - r_i >= 0
        cons.extend(h - ys - rs)
        
        # Disjointness constraints:
        # dist(i,j)^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        
        # Create index arrays for pairs i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist_sq = dx**2 + dy**2
        
        sum_r = rs[i_idx] + rs[j_idx]
        sum_r_sq = sum_r**2
        
        cons.extend(dist_sq - sum_r_sq)
        
        return cons

    def objective(v):
        idx_r = np.arange(3, 3*N + 1, 3)
        rs = v[idx_r]
        return -np.sum(rs)

    def bounds():
        b = [(0.1, 1.9)]  # w
        for i in range(N):
            b.append((0.0, 2.0)) # x
            b.append((0.0, 2.0)) # y
            b.append((0.01, 1.0)) # r
        return b

    best_sol = get_fallback_solution(1.0)
    best_score = np.sum(best_sol[:, 2])
    
    # Try different aspect ratios
    aspect_ratios = [0.8, 1.0, 1.2]
    
    for w_init in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        h_init = 2.0 - w_init
        
        # Initialize with a grid
        cols = 5
        rows = 5
        v0 = np.zeros(1 + 3*N)
        v0[0] = w_init
        
        # Generate grid points
        xs_grid = np.linspace(w_init / (2*cols), w_init - w_init / (2*cols), cols)
        ys_grid = np.linspace(h_init / (2*rows), h_init - h_init / (2*rows), rows)
        
        X, Y = np.meshgrid(xs_grid, ys_grid)
        X = X.flatten()
        Y = Y.flatten()
        
        # Take first N points
        X = X[:N]
        Y = Y[:N]
        
        # Initial radius
        dx = w_init / cols
        dy = h_init / rows
        r_init = min(dx, dy) * 0.49
        
        for k in range(N):
            v0[1 + 3*k] = X[k]
            v0[2 + 3*k] = Y[k]
            v0[3 + 3*k] = r_init
            
        # Run optimization
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds(), 
                           constraints=[{'type': 'ineq', 'fun': constraints}],
                           options={'maxiter': 100, 'ftol': 1e-7, 'disp': False})
            
            if res.success:
                v_opt = res.x
                w_opt = v_opt[0]
                
                # Extract solution
                idx_x = np.arange(1, 3*N + 1, 3)
                idx_y = np.arange(2, 3*N + 1, 3)
                idx_r = np.arange(3, 3*N + 1, 3)
                
                xs = v_opt[idx_x]
                ys = v_opt[idx_y]
                rs = v_opt[idx_r]
                
                sol = np.column_stack((xs, ys, rs))
                score = np.sum(rs)
                
                # Strict validity check
                h_opt = 2.0 - w_opt
                valid = True
                eps = 1e-6
                
                # Boundary
                if np.any(xs < rs - eps) or np.any(w_opt - xs < rs - eps) or \
                   np.any(ys < rs - eps) or np.any(h_opt - ys < rs - eps):
                    valid = False
                
                # Overlap
                if valid:
                    i_idx, j_idx = np.triu_indices(N, k=1)
                    dx = xs[i_idx] - xs[j_idx]
                    dy = ys[i_idx] - ys[j_idx]
                    dist_sq = dx**2 + dy**2
                    sum_r_sq = (rs[i_idx] + rs[j_idx])**2
                    if np.any(dist_sq < sum_r_sq - eps):
                        valid = False
                
                if valid and score > best_score:
                    best_score = score
                    best_sol = sol
                    
        except Exception as e:
            # If optimizer crashes, keep previous best
            pass

    # Final check: ensure we return a valid array
    # If best_sol is somehow invalid (shouldn't happen with fallback), use fallback
    # But get_fallback_solution is guaranteed valid for small radii.
    
    return best_sol

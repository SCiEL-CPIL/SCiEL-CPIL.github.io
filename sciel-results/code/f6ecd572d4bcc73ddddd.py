import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    # Helper to generate a hexagonal grid
    def hex_grid(n, W, H):
        # Generate points in a hexagonal pattern scaled to W x H
        # Approximate number of rows/cols
        side = int(np.ceil(np.sqrt(n * H / W)))
        x = []
        y = []
        for row in range(side + 2):
            for col in range(side + 2):
                px = (col + 0.5 * (row % 2)) * (W / (side + 1))
                py = row * (H / (side + 1)) * 0.866
                if px < W and py < H:
                    x.append(px)
                    y.append(py)
        
        # If we have more than n, take the first n (or random sample if we want diversity, but deterministic is safer)
        # To ensure we get n points, we might need to adjust the grid density
        # Let's just generate a dense grid and pick n points that are well spread
        # For simplicity, we'll just use the first n if enough, else pad
        if len(x) < n:
             # Fallback to square if hex is too sparse
             side_sq = int(np.ceil(np.sqrt(n)))
             x = [(i + 0.5) * W / side_sq for i in range(side_sq) for _ in range(side_sq)][:n]
             y = [(j + 0.5) * H / side_sq for _ in range(side_sq) for j in range(side_sq)][:n]
        else:
             x = x[:n]
             y = y[:n]
        return np.array(x), np.array(y)

    def evaluate_config(x, y, r, W, H):
        if W <= 0 or H <= 0:
            return -np.inf
        if abs(2*(W+H) - 4) > 1e-6:
            return -np.inf
        for i in range(N):
            if r[i] <= 0:
                return -np.inf
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > W + 1e-6:
                return -np.inf
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > H + 1e-6:
                return -np.inf
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-6:
                    return -np.inf
        return np.sum(r)

    def greedy_expand(x, y, r, W, H):
        # Simple greedy radius expansion
        step = 0.01
        for iteration in range(50):
            improved = False
            for i in range(N):
                new_r = r[i] + step
                if new_r <= 0:
                    continue
                if x[i] - new_r < 0 or x[i] + new_r > W:
                    continue
                if y[i] - new_r < 0 or y[i] + new_r > H:
                    continue
                valid = True
                for j in range(N):
                    if i == j: continue
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    rad_sum = new_r + r[j]
                    if dist_sq < rad_sum**2:
                        valid = False
                        break
                if valid:
                    r[i] = new_r
                    improved = True
            if not improved:
                step *= 0.8
                if step < 1e-4:
                    break
        return r

    def cyclic_coordinate_descent(x, y, r, W, H, iterations=20):
        """
        Optimizes each circle's (x, y, r) sequentially, fixing the others.
        This mimics the 'autoregressive' or sequential placement improvement.
        """
        for _ in range(iterations):
            for i in range(N):
                # Define objective for circle i: maximize r_i
                # Variables: x_i, y_i, r_i
                # Constraints:
                # 1. x_i - r_i >= 0
                # 2. x_i + r_i <= W
                # 3. y_i - r_i >= 0
                # 4. y_i + r_i <= H
                # 5. dist(i, j) >= r_i + r_j for all j != i
                # 6. r_i > 0
                
                def obj_ci(v):
                    return -v[2] # minimize -r_i
                
                def jac_ci(v):
                    return np.array([0.0, 0.0, -1.0])
                
                cons_ci = []
                # Wall constraints
                cons_ci.append({'type': 'ineq', 'fun': lambda v: v[0] - v[2]})
                cons_ci.append({'type': 'ineq', 'fun': lambda v: W - v[0] - v[2]})
                cons_ci.append({'type': 'ineq', 'fun': lambda v: v[1] - v[2]})
                cons_ci.append({'type': 'ineq', 'fun': lambda v: H - v[1] - v[2]})
                cons_ci.append({'type': 'ineq', 'fun': lambda v: v[2]})
                
                # Interaction constraints with other circles
                for j in range(N):
                    if i == j: continue
                    xj, yj, rj = x[j], y[j], r[j]
                    # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
                    # This is non-convex. SLSQP can handle it locally.
                    cons_ci.append({
                        'type': 'ineq', 
                        'fun': lambda v, j=j: (v[0]-xj)**2 + (v[1]-yj)**2 - (v[2]+rj)**2
                    })
                
                v0 = np.array([x[i], y[i], r[i]])
                try:
                    res = minimize(obj_ci, v0, jac=jac_ci, method='SLSQP', constraints=cons_ci, options={'maxiter': 50, 'ftol': 1e-6})
                    if res.success and res.x[2] > r[i]:
                        x[i] = res.x[0]
                        y[i] = res.x[1]
                        r[i] = res.x[2]
                except:
                    pass
        return x, y, r

    # Generate initial candidates with different aspect ratios
    candidates = []
    
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66]
    
    for ar in aspect_ratios:
        # W + H = 2
        # W / H = ar => W = ar * H => H(ar + 1) = 2 => H = 2 / (ar + 1), W = 2 * ar / (ar + 1)
        H = 2.0 / (ar + 1)
        W = 2.0 * ar / (ar + 1)
        
        # Hex grid init
        x_hex, y_hex = hex_grid(N, W, H)
        r_hex = np.full(N, 0.02)
        r_hex = greedy_expand(x_hex, y_hex, r_hex, W, H)
        score_hex = evaluate_config(x_hex, y_hex, r_hex, W, H)
        candidates.append((x_hex, y_hex, r_hex, W, H, score_hex, 'hex'))
        
        # Square grid init
        side = int(np.ceil(np.sqrt(N)))
        x_sq = np.array([(i + 0.5) * W / side for i in range(side) for _ in range(side)][:N])
        y_sq = np.array([(j + 0.5) * H / side for _ in range(side) for j in range(side)][:N])
        r_sq = np.full(N, 0.02)
        r_sq = greedy_expand(x_sq, y_sq, r_sq, W, H)
        score_sq = evaluate_config(x_sq, y_sq, r_sq, W, H)
        candidates.append((x_sq, y_sq, r_sq, W, H, score_sq, 'sq'))

    # Sort by score
    candidates.sort(key=lambda c: c[5], reverse=True)
    
    best_x, best_y, best_r, best_W, best_H, best_score, _ = candidates[0]
    
    # Run Cyclic Coordinate Descent on the best candidate
    # This is the "autoregressive" improvement step
    best_x, best_y, best_r = cyclic_coordinate_descent(best_x, best_y, best_r, best_W, best_H, iterations=10)
    new_score = evaluate_config(best_x, best_y, best_r, best_W, best_H)
    if new_score > best_score:
        best_score = new_score

    # Run a final global SLSQP to polish the sum of radii
    def obj(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        return -np.sum(r)
        
    def jac(v):
        g = np.zeros_like(v)
        g[2*N:3*N] = -1.0
        return g
        
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
    
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
            })
            
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
        
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N]})
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N+1]})

    x_start = np.concatenate([best_x, best_y, best_r, [best_W, best_H]])
    
    try:
        res = minimize(obj, x_start, jac=jac, method='SLSQP', constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
        if res.success:
            x_final = res.x[:N]
            y_final = res.x[N:2*N]
            r_final = res.x[2*N:3*N]
            W_final = res.x[3*N]
            H_final = res.x[3*N+1]
            
            score_final = evaluate_config(x_final, y_final, r_final, W_final, H_final)
            if score_final > best_score:
                best_x, best_y, best_r, best_W, best_H = x_final, y_final, r_final, W_final, H_final
                best_score = score_final
    except Exception:
        pass

    # Final check and return
    if best_score == -np.inf:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
        r = np.full(N, 0.5 / side - 0.01)
        W, H = 1.0, 1.0
        return np.column_stack([x, y, r])
        
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result

import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    def get_max_radius(x, y, circles, W, H):
        r_max = min(x, W - x, y, H - y)
        if r_max <= 0:
            return 0.0
        for cx, cy, cr in circles:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            r_max = min(r_max, dist - cr)
        return max(0.0, r_max)

    def place_one_circle(circles, W, H):
        grid_size = 20
        x_grid = np.linspace(0.05, W - 0.05, grid_size)
        y_grid = np.linspace(0.05, H - 0.05, grid_size)
        
        best_x, best_y, best_r = 0, 0, 0
        
        for gx in x_grid:
            for gy in y_grid:
                r = get_max_radius(gx, gy, circles, W, H)
                if r > best_r:
                    best_r = r
                    best_x = gx
                    best_y = gy
        
        if best_r <= 0:
            return None
            
        def obj(v):
            return -v[2]
            
        def jac(v):
            return np.array([0, 0, -1])
            
        constraints = []
        constraints.append({'type': 'ineq', 'fun': lambda v: v[0] - v[2]})
        constraints.append({'type': 'ineq', 'fun': lambda v: W - v[0] - v[2]})
        constraints.append({'type': 'ineq', 'fun': lambda v: v[1] - v[2]})
        constraints.append({'type': 'ineq', 'fun': lambda v: H - v[1] - v[2]})
        
        for cx, cy, cr in circles:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, cx=cx, cy=cy, cr=cr: (v[0]-cx)**2 + (v[1]-cy)**2 - (v[2]+cr)**2
            })
            
        constraints.append({'type': 'ineq', 'fun': lambda v: v[2] - 0.01})
        
        x0 = np.array([best_x, best_y, best_r])
        
        try:
            res = minimize(obj, x0, jac=jac, method='SLSQP', constraints=constraints, options={'maxiter': 50, 'ftol': 1e-6})
            if res.success:
                x_final, y_final, r_final = res.x
                if r_final > 0 and get_max_radius(x_final, y_final, circles, W, H) >= r_final - 1e-5:
                    return (x_final, y_final, r_final)
        except Exception:
            pass
            
        return (best_x, best_y, best_r)

    def autoregressive_pack(W, H, n_circles=N):
        circles = []
        for _ in range(n_circles):
            new_circle = place_one_circle(circles, W, H)
            if new_circle is None:
                break
            circles.append(new_circle)
        return circles

    def evaluate_circles(circles, W, H):
        total_r = 0
        for x, y, r in circles:
            if r <= 0:
                return -np.inf
            if x - r < -1e-6 or x + r > W + 1e-6 or y - r < -1e-6 or y + r > H + 1e-6:
                return -np.inf
            total_r += r
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                rad_sum = r1 + r2
                if dist_sq < rad_sum**2 - 1e-6:
                    return -np.inf
        return total_r

    best_score = -np.inf
    best_circles = None
    best_W, best_H = 1.0, 1.0
    
    aspects = [1.0, 1.2, 0.8]
    
    for aspect in aspects:
        W = 2.0 / (1.0 + aspect)
        H = 2.0 - W
        
        circles = autoregressive_pack(W, H)
        score = evaluate_circles(circles, W, H)
        
        if score > best_score:
            best_score = score
            best_circles = circles
            best_W, best_H = W, H
            
    if best_circles is None or best_score == -np.inf:
        W, H = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
        r = np.full(N, 0.5 / side - 0.01)
        return np.column_stack([x, y, r])
        
    # Global Polish
    if len(best_circles) == N:
        x_arr = np.array([c[0] for c in best_circles])
        y_arr = np.array([c[1] for c in best_circles])
        r_arr = np.array([c[2] for c in best_circles])
        
        def obj(v):
            return -np.sum(v[2*N:3*N])
            
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
        
        x_start = np.concatenate([x_arr, y_arr, r_arr, [best_W, best_H]])
        
        try:
            res = minimize(obj, x_start, jac=jac, method='SLSQP', constraints=constraints, options={'maxiter': 300, 'ftol': 1e-8})
            if res.success:
                x_final = res.x[:N]
                y_final = res.x[N:2*N]
                r_final = res.x[2*N:3*N]
                W_final = res.x[3*N]
                H_final = res.x[3*N+1]
                
                circles_final = list(zip(x_final, y_final, r_final))
                score_final = evaluate_circles(circles_final, W_final, H_final)
                if score_final > best_score:
                    best_score = score_final
                    best_circles = circles_final
                    best_W, best_H = W_final, H_final
        except Exception:
            pass
            
    result = np.zeros((N, 3))
    for i in range(len(best_circles)):
        x, y, r = best_circles[i]
        result[i, 0] = x
        result[i, 1] = y
        result[i, 2] = r
        
    return result

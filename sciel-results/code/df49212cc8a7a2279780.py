import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_value = -np.inf
    best_solution = None
    
    def cost_and_grad(z):
        w = z[0]
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return 1e10, np.zeros_like(z)
        
        x = z[1::3]
        y = z[2::3]
        r = z[3::3]
        
        cost = -np.sum(r)
        grad = np.zeros_like(z)
        
        Lambda = 20.0
        
        for i in range(N):
            xi, yi, ri = x[i], y[i], r[i]
            
            viol_l = ri - xi
            if viol_l > 0:
                cost += Lambda * viol_l**2
                grad[1 + 3*i] += -2 * Lambda * viol_l
                grad[3 + 3*i] += 2 * Lambda * viol_l
            
            viol_r = xi + ri - w
            if viol_r > 0:
                cost += Lambda * viol_r**2
                grad[1 + 3*i] += 2 * Lambda * viol_r
                grad[3 + 3*i] += 2 * Lambda * viol_r
                grad[0] += -2 * Lambda * viol_r
            
            viol_b = ri - yi
            if viol_b > 0:
                cost += Lambda * viol_b**2
                grad[2 + 3*i] += -2 * Lambda * viol_b
                grad[3 + 3*i] += 2 * Lambda * viol_b
            
            viol_t = yi + ri - (2.0 - w)
            if viol_t > 0:
                cost += Lambda * viol_t**2
                grad[2 + 3*i] += 2 * Lambda * viol_t
                grad[3 + 3*i] += 2 * Lambda * viol_t
                grad[0] += 2 * Lambda * viol_t
                
            grad[3 + 3*i] += -1.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                r_sum = r[i] + r[j]
                
                viol = r_sum - dist
                if viol > 0:
                    cost += Lambda * viol**2
                    grad[1 + 3*i] += 2 * Lambda * viol * ( -dx / dist )
                    grad[2 + 3*i] += 2 * Lambda * viol * ( -dy / dist )
                    grad[1 + 3*j] += 2 * Lambda * viol * ( dx / dist )
                    grad[2 + 3*j] += 2 * Lambda * viol * ( dy / dist )
                    grad[3 + 3*i] += 2 * Lambda * viol
                    grad[3 + 3*j] += 2 * Lambda * viol

        return cost, grad

    def discrete_init(w, h, grid_res=20):
        xs = np.linspace(0, w, grid_res)
        ys = np.linspace(0, h, grid_res)
        
        candidates = []
        for i in range(grid_res):
            for j in range(grid_res):
                cx = xs[i]
                cy = ys[j]
                max_r = min(cx, w - cx, cy, h - cy)
                if max_r > 0.01:
                    candidates.append((max_r, cx, cy))
        
        candidates.sort(reverse=True)
        
        selected = []
        for max_r, cx, cy in candidates:
            r_try = max_r * 0.95
            is_valid = True
            for (sx, sy, sr) in selected:
                dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)
                if dist < r_try + sr:
                    is_valid = False
                    break
            
            if is_valid:
                selected.append((cx, cy, r_try))
                if len(selected) == N:
                    break
        
        if len(selected) < N:
            return None
            
        z0 = np.zeros(1 + 3*N)
        z0[0] = w
        for i in range(N):
            cx, cy, r = selected[i]
            z0[1 + 3*i] = cx
            z0[2 + 3*i] = cy
            z0[3 + 3*i] = r
        return z0

    aspect_ratios = np.linspace(0.6, 1.4, 10)
    
    for w in aspect_ratios:
        h = 2.0 - w
        if h < 0.1:
            continue
            
        z0 = discrete_init(w, h)
        if z0 is None:
            continue
            
        try:
            res = minimize(cost_and_grad, z0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 300, 'ftol': 1e-8})
            
            z_opt = res.x
            w_opt = z_opt[0]
            h_opt = 2.0 - w_opt
            
            if w_opt <= 0.1 or h_opt <= 0.1:
                continue
                
            x = z_opt[1::3]
            y = z_opt[2::3]
            r = z_opt[3::3]
            
            valid = True
            if np.any(r <= 0):
                valid = False
            
            if valid:
                for i in range(N):
                    if x[i] < r[i] - 1e-5 or x[i] > w_opt - r[i] + 1e-5 or y[i] < r[i] - 1e-5 or y[i] > h_opt - r[i] + 1e-5:
                        valid = False
                        break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r[i] + r[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_solution = np.column_stack((x, y, r))
                    
        except Exception:
            continue

    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution

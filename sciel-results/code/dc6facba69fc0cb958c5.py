import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Define loss function and gradient
    def loss_and_grad(v, lam_ov=10.0, lam_box=10.0, delta=1e-4):
        # v shape (3N,)
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        loss = -np.sum(r)
        grad = np.zeros_like(v)
        grad[2::3] -= 1.0  # d(-sum r)/dr = -1
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq + delta*delta)
                
                dist_sum = r[i] + r[j]
                diff = dist_sum - d
                
                if diff > 0:
                    # Smooth approximation of max(0, diff)^2
                    # Actually, let's use the smooth function directly in the loss
                    # P = 0.5 * (diff + sqrt(diff^2 + delta^2))^2 ? No, that's for max(0,diff).
                    # Let's use softplus for the linear term and square it?
                    # Simpler: Use the exact penalty but with a smooth sqrt in distance.
                    # The kink is at diff=0.
                    # Let's use a smooth penalty: P = 0.5 * (diff + sqrt(diff^2 + eps^2))^2
                    # This is smooth.
                    s = np.sqrt(diff*diff + delta*delta)
                    p = 0.5 * (diff + s)
                    p_sq = p * p
                    
                    loss += lam_ov * p_sq
                    
                    # Gradient of p_sq = 2 * p * dp/ddiff * ddiff/dvar
                    # dp/ddiff = 0.5 * (1 + diff/s)
                    dp_ddiff = 0.5 * (1.0 + diff / s)
                    d_p_sq = 2.0 * p * dp_ddiff
                    
                    # ddiff/dr_i = 1, ddiff/dr_j = 1
                    grad[2*i] += lam_ov * d_p_sq
                    grad[2*j] += lam_ov * d_p_sq
                    
                    # ddiff/dx_i = -dx/d, ddiff/dy_i = -dy/d
                    # ddiff/dx_j = dx/d, ddiff/dy_j = dy/d
                    dx_dd = dx / d
                    dy_dd = dy / d
                    
                    grad[i] += lam_ov * d_p_sq * (-dx_dd)
                    grad[i+1] += lam_ov * d_p_sq * (-dy_dd)
                    grad[j] += lam_ov * d_p_sq * (dx_dd)
                    grad[j+1] += lam_ov * d_p_sq * (dy_dd)
        
        # Boundary penalties
        # L: x < r -> diff = r - x
        # R: 1-x < r -> diff = r - (1-x) = r + x - 1
        # B: y < r -> diff = r - y
        # T: 1-y < r -> diff = r - (1-y) = r + y - 1
        
        for i in range(N):
            # Left
            diff_l = r[i] - x[i]
            if diff_l > 0:
                s = np.sqrt(diff_l*diff_l + delta*delta)
                p = 0.5 * (diff_l + s)
                p_sq = p * p
                loss += lam_box * p_sq
                dp_ddiff = 0.5 * (1.0 + diff_l / s)
                d_p_sq = 2.0 * p * dp_ddiff
                grad[2*i] += lam_box * d_p_sq # dr
                grad[i] -= lam_box * d_p_sq    # dx
            # Right
            diff_r = r[i] + x[i] - 1.0
            if diff_r > 0:
                s = np.sqrt(diff_r*diff_r + delta*delta)
                p = 0.5 * (diff_r + s)
                p_sq = p * p
                loss += lam_box * p_sq
                dp_ddiff = 0.5 * (1.0 + diff_r / s)
                d_p_sq = 2.0 * p * dp_ddiff
                grad[2*i] += lam_box * d_p_sq # dr
                grad[i] += lam_box * d_p_sq    # dx
            # Bottom
            diff_b = r[i] - y[i]
            if diff_b > 0:
                s = np.sqrt(diff_b*diff_b + delta*delta)
                p = 0.5 * (diff_b + s)
                p_sq = p * p
                loss += lam_box * p_sq
                dp_ddiff = 0.5 * (1.0 + diff_b / s)
                d_p_sq = 2.0 * p * dp_ddiff
                grad[2*i] += lam_box * d_p_sq # dr
                grad[i+1] -= lam_box * d_p_sq # dy
            # Top
            diff_t = r[i] + y[i] - 1.0
            if diff_t > 0:
                s = np.sqrt(diff_t*diff_t + delta*delta)
                p = 0.5 * (diff_t + s)
                p_sq = p * p
                loss += lam_box * p_sq
                dp_ddiff = 0.5 * (1.0 + diff_t / s)
                d_p_sq = 2.0 * p * dp_ddiff
                grad[2*i] += lam_box * d_p_sq # dr
                grad[i+1] += lam_box * d_p_sq # dy
        
        return loss, grad

    def get_score(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # Check validity
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > 1-r[i] + 1e-9 or y[i] < r[i] - 1e-9 or y[i] > 1-r[i] + 1e-9:
                valid = False
                break
        if not valid:
            return -np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return -np.inf
        return np.sum(r)

    # Initializations
    inits = []
    
    # Grid 5x5 (25 circles) + 1 random
    # Actually, let's just use random and grid.
    
    # 1. Grid 5x5, take first 26? No, 5x5 is 25.
    # Let's create a 5x5 grid and add one in the center of the largest hole?
    # Simpler: 4x7 grid = 28. Remove 2.
    # Or just random.
    
    rng = np.random.default_rng(42)
    
    # Random inits
    for _ in range(10):
        x = rng.uniform(0.2, 0.8, N)
        y = rng.uniform(0.2, 0.8, N)
        r = rng.uniform(0.05, 0.1, N)
        v = np.zeros(3*N)
        v[0::3] = x
        v[1::3] = y
        v[2::3] = r
        inits.append(v)
        
    # Grid init
    # 5 rows, 5 cols = 25. Add 1 more.
    # Let's try a hexagonal packing init.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle?
    # Total 26.
    
    # Let's just use a few different grid spacings.
    for rows in [4, 5, 6]:
        cols = int(np.ceil(N / rows))
        # If cols * rows > N, we have extra spots.
        # We will fill row by row.
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) / cols
                y[idx] = (i + 0.5) / rows
                r[idx] = 0.5 / max(rows, cols) * 0.9
                idx += 1
        if idx < N:
            # Fill remaining with random small circles
            for k in range(idx, N):
                x[k] = rng.uniform(0.2, 0.8)
                y[k] = rng.uniform(0.2, 0.8)
                r[k] = 0.05
        v = np.zeros(3*N)
        v[0::3] = x
        v[1::3] = y
        v[2::3] = r
        inits.append(v)

    # Optimization loop
    for v0 in inits:
        if time.time() - start_time > time_limit:
            break
            
        # Use L-BFGS-B
        # Bounds: x,y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # We can try different lambda values
        for lam in [1.0, 10.0, 100.0]:
            if time.time() - start_time > time_limit:
                break
                
            def obj(v):
                l, g = loss_and_grad(v, lam_ov=lam, lam_box=lam)
                return l
            
            def grad(v):
                l, g = loss_and_grad(v, lam_ov=lam, lam_box=lam)
                return g
                
            try:
                res = minimize(obj, v0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6})
            except Exception:
                continue
                
            v_res = res.x
            score = get_score(v_res)
            
            if score > best_score:
                best_score = score
                best_result = v_res.copy()
                
    # If no valid result found, return a trivial valid one
    if best_result is None:
        # Return a grid of small circles
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Reshape to (N, 3)
    result = best_result.reshape(N, 3)
    return result

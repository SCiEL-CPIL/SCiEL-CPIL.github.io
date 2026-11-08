import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_result = None
    best_score = -np.inf
    
    # Try a range of aspect ratios W
    # Perimeter 4 => 2(W+H)=4 => W+H=2. Let W be width, H be height.
    # W can range from small to large. By symmetry, W in [0, 2].
    # Let's sample W values.
    w_values = np.linspace(0.5, 1.5, 11)
    
    # Initial configurations
    def get_initial_config(w, h, seed):
        np.random.seed(seed)
        # Grid-like initialization
        side = int(np.ceil(np.sqrt(N)))
        # Adjust grid to fit W and H
        # Simple random initialization might be better to escape local minima
        xs = np.random.uniform(0.1*w, 0.9*w, N)
        ys = np.random.uniform(0.1*h, 0.9*h, N)
        # Initial radii: small
        rs = np.full(N, 0.05 * min(w, h))
        # Pack into vector
        x = np.concatenate([xs, ys, rs, [w]])
        return x

    def objective_and_grad(x):
        # Unpack
        xs = x[0:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        w = x[3*N]
        h = 2.0 - w
        
        # Objective: -sum(r)
        val = -np.sum(rs)
        
        # Penalties
        pen = 0.0
        
        # Pairwise constraints: r_i + r_j <= dist(i, j)
        # dist(i, j) = sqrt((xi-xj)^2 + (yi-yj)^2)
        # To avoid sqrt cost in gradient, we can use squared distance?
        # Constraint: (ri + rj)^2 <= (xi-xj)^2 + (yi-yj)^2
        # Let g = (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2
        # We want g >= 0. Penalty max(0, -g)^2?
        # This is smoother? No, max(0, -g) is still kinked.
        # Let's stick to distance for clarity, N=21 is small (210 pairs).
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    pen += overlap * overlap
        
        # Boundary constraints
        for i in range(N):
            # x bounds
            o1 = rs[i] - xs[i]
            if o1 > 0: pen += o1 * o1
            o2 = rs[i] - (w - xs[i])
            if o2 > 0: pen += o2 * o2
            
            # y bounds
            o3 = rs[i] - ys[i]
            if o3 > 0: pen += o3 * o3
            o4 = rs[i] - (h - ys[i])
            if o4 > 0: pen += o4 * o4
            
        return val + 100.0 * pen # Weight penalty

    # We need gradients for L-BFGS-B to be fast.
    # Let's implement analytical gradients.
    
    def objective_grad(x):
        xs = x[0:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        w = x[3*N]
        h = 2.0 - w
        
        grad = np.zeros_like(x)
        
        # Gradient of -sum(r)
        grad[2*N:3*N] = -1.0
        
        pen = 0.0
        dpen = np.zeros_like(x)
        
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    # d/dx (overlap^2) = 2 * overlap * d/dx (overlap)
                    # d/dx_i (overlap) = -dx/d
                    # d/dx_j (overlap) = dx/d
                    # d/dr_i (overlap) = 1
                    # d/dr_j (overlap) = 1
                    
                    factor = 2.0 * overlap
                    
                    # x_i
                    dpen[0:N][i] += factor * (-dx/d)
                    # y_i
                    dpen[N:2*N][i] += factor * (-dy/d)
                    # r_i
                    dpen[2*N:3*N][i] += factor * 1.0
                    
                    # x_j
                    dpen[0:N][j] += factor * (dx/d)
                    # y_j
                    dpen[N:2*N][j] += factor * (dy/d)
                    # r_j
                    dpen[2*N:3*N][j] += factor * 1.0
        
        # Boundary
        for i in range(N):
            # x_left: r_i - x_i
            o1 = rs[i] - xs[i]
            if o1 > 0:
                f = 2.0 * o1
                dpen[0:N][i] += f * (-1.0) # d/dx_i
                dpen[2*N:3*N][i] += f * (1.0)  # d/dr_i
                
            # x_right: r_i - (w - x_i) = r_i - w + x_i
            o2 = rs[i] - (w - xs[i])
            if o2 > 0:
                f = 2.0 * o2
                dpen[0:N][i] += f * (1.0)  # d/dx_i
                dpen[2*N:3*N][i] += f * (1.0)  # d/dr_i
                dpen[3*N] += f * (-1.0)   # d/dw : d/dw (r_i - w + x_i) = -1
                
            # y_bottom: r_i - y_i
            o3 = rs[i] - ys[i]
            if o3 > 0:
                f = 2.0 * o3
                dpen[N:2*N][i] += f * (-1.0) # d/dy_i
                dpen[2*N:3*N][i] += f * (1.0)  # d/dr_i
                
            # y_top: r_i - (h - y_i) = r_i - (2-w) + y_i = r_i - 2 + w + y_i
            o4 = rs[i] - (h - ys[i])
            if o4 > 0:
                f = 2.0 * o4
                dpen[N:2*N][i] += f * (1.0)  # d/dy_i
                dpen[2*N:3*N][i] += f * (1.0)  # d/dr_i
                dpen[3*N] += f * (1.0)    # d/dw : d/dw (r_i - 2 + w + y_i) = 1
        
        pen = 100.0 * (np.sum(dpen**2)/2.0) # This is not right, pen is sum of squares.
        # Actually, we just add the gradient of the penalty term.
        # Penalty term was sum(overlap^2). Grad is dpen.
        # Total objective = -sum(r) + 100 * Penalty.
        # Grad = grad_obj + 100 * dpen.
        
        grad += 100.0 * dpen
        
        return -np.sum(rs) + 100.0 * (pen_val), grad

    # Wait, I need to calculate pen_val correctly in the function above.
    # Let's refactor to return value and gradient together.

    def objective_and_grad_impl(x):
        xs = x[0:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        w = x[3*N]
        h = 2.0 - w
        
        val = -np.sum(rs)
        dpen = np.zeros_like(x)
        pen_val = 0.0
        
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    pen_val += overlap * overlap
                    factor = 2.0 * overlap
                    dpen[0:N][i] += factor * (-dx/d)
                    dpen[N:2*N][i] += factor * (-dy/d)
                    dpen[2*N:3*N][i] += factor * 1.0
                    dpen[0:N][j] += factor * (dx/d)
                    dpen[N:2*N][j] += factor * (dy/d)
                    dpen[2*N:3*N][j] += factor * 1.0
        
        # Boundary
        for i in range(N):
            o1 = rs[i] - xs[i]
            if o1 > 0:
                pen_val += o1 * o1
                f = 2.0 * o1
                dpen[0:N][i] += f * (-1.0)
                dpen[2*N:3*N][i] += f * (1.0)
                
            o2 = rs[i] - (w - xs[i])
            if o2 > 0:
                pen_val += o2 * o2
                f = 2.0 * o2
                dpen[0:N][i] += f * (1.0)
                dpen[2*N:3*N][i] += f * (1.0)
                dpen[3*N] += f * (-1.0)
                
            o3 = rs[i] - ys[i]
            if o3 > 0:
                pen_val += o3 * o3
                f = 2.0 * o3
                dpen[N:2*N][i] += f * (-1.0)
                dpen[2*N:3*N][i] += f * (1.0)
                
            o4 = rs[i] - (h - ys[i])
            if o4 > 0:
                pen_val += o4 * o4
                f = 2.0 * o4
                dpen[N:2*N][i] += f * (1.0)
                dpen[2*N:3*N][i] += f * (1.0)
                dpen[3*N] += f * (1.0)
        
        val += 100.0 * pen_val
        grad = np.zeros_like(x)
        grad[2*N:3*N] = -1.0
        grad += 100.0 * dpen
        
        return val, grad

    # Bounds
    # We need to update bounds if W changes? 
    # L-BFGS-B requires fixed bounds.
    # If W is a variable, the bounds for x_i depend on W.
    # This is tricky for standard L-BFGS-B.
    # Solution: Transform variables to be in [0,1] and scale by W and H.
    # u_i in [0,1], v_i in [0,1].
    # x_i = u_i * W, y_i = v_i * H.
    # Then bounds for u, v are [0,1].
    # r_i bounds [0, 1].
    # W bounds [0.5, 1.5].
    
    def objective_transformed(x):
        us = x[0:N]
        vs = x[N:2*N]
        rs = x[2*N:3*N]
        w = x[3*N]
        h = 2.0 - w
        
        xs = us * w
        ys = vs * h
        
        val = -np.sum(rs)
        dpen_u = np.zeros_like(x)
        pen_val = 0.0
        
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    pen_val += overlap * overlap
                    factor = 2.0 * overlap
                    # d/dx_i = -dx/d, d/dx_j = dx/d
                    # x_i = u_i * w => d/dx_i = (1/w) d/du_i
                    # y_i = v_i * h => d/dy_i = (1/h) d/dv_i
                    
                    dpen_u[0:N][i] += factor * (-dx/d) * (1.0/w)
                    dpen_u[N:2*N][i] += factor * (-dy/d) * (1.0/h)
                    dpen_u[2*N:3*N][i] += factor * 1.0
                    dpen_u[0:N][j] += factor * (dx/d) * (1.0/w)
                    dpen_u[N:2*N][j] += factor * (dy/d) * (1.0/h)
                    dpen_u[2*N:3*N][j] += factor * 1.0
                    # Gradient wrt w and h
                    # d/dw (overlap^2) = 2*overlap * d/dw(overlap)
                    # overlap = r_i + r_j - sqrt((u_i w - u_j w)^2 + (v_i h - v_j h)^2)
                    # This is complex. Let's just compute numerical gradient for w?
                    # Or derive:
                    # d/dw (r_i + r_j - d) = - d/dw d
                    # d = sqrt(w^2(u_i-u_j)^2 + h^2(v_i-v_j)^2)
                    # d/dw d = (1/(2d)) * 2 w^2 (u_i-u_j)^2 = w^2 (u_i-u_j)^2 / d
                    # h = 2-w, dh/dw = -1
                    # d/dh d = h^2 (v_i-v_j)^2 / d
                    # d/dw d = d/dw ( ... ) + d/dh ( ... ) * (-1)
                    # = w^2 (u_i-u_j)^2 / d - h^2 (v_i-v_j)^2 / d
                    #
                    # This is doable but error prone. 
                    # Given N=21, maybe we can just use finite difference for w?
                    # Or simply bound w and not optimize it, but try multiple w values.
                    # Since we are doing restarts, we can fix w in each run.
                    # This simplifies the gradient calculation significantly.
                    pass
        
        # If we fix w, we don't need gradient for w.
        # Let's restart the strategy:
        # Loop over w values.
        # For each w, optimize u, v, r.
        
        # I will rewrite the loop structure to fix w.
        
        return val, dpen_u # Placeholder

    # Revised Plan:
    # 1. Loop w in linspace(0.5, 1.5, 15)
    # 2. For each w, run L-BFGS-B on variables [u, v, r] with bounds [0,1] for u,v and [0,1] for r.
    # 3. Objective: -sum(r) + penalty.
    # 4. Gradient: analytical for u, v, r.
    
    def run_optimization_for_w(w, seed):
        h = 2.0 - w
        np.random.seed(seed)
        # Initial guess
        u0 = np.random.uniform(0.1, 0.9, N)
        v0 = np.random.uniform(0.1, 0.9, N)
        r0 = np.full(N, 0.1 * min(w, h))
        x0 = np.concatenate([u0, v0, r0])
        
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 1)] * N
        
        def obj_grad(x):
            us = x[0:N]
            vs = x[N:2*N]
            rs = x[2*N:3*N]
            
            xs = us * w
            ys = vs * h
            
            val = -np.sum(rs)
            dpen = np.zeros_like(x)
            pen_val = 0.0
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d_sq = dx*dx + dy*dy
                    d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
                    overlap = rs[i] + rs[j] - d
                    if overlap > 0:
                        pen_val += overlap * overlap
                        factor = 2.0 * overlap
                        # d/dx_i = -dx/d, d/dx_j = dx/d
                        # d/dx_i = (1/w) d/du_i
                        dpen[0:N][i] += factor * (-dx/d) * (1.0/w)
                        dpen[N:2*N][i] += factor * (-dy/d) * (1.0/h)
                        dpen[2*N:3*N][i] += factor * 1.0
                        dpen[0:N][j] += factor * (dx/d) * (1.0/w)
                        dpen[N:2*N][j] += factor * (dy/d) * (1.0/h)
                        dpen[2*N:3*N][j] += factor * 1.0
            
            for i in range(N):
                o1 = rs[i] - xs[i]
                if o1 > 0:
                    pen_val += o1 * o1
                    f = 2.0 * o1
                    dpen[0:N][i] += f * (-1.0/w)
                    dpen[2*N:3*N][i] += f * 1.0
                    
                o2 = rs[i] - (w - xs[i])
                if o2 > 0:
                    pen_val += o2 * o2
                    f = 2.0 * o2
                    dpen[0:N][i] += f * (1.0/w)
                    dpen[2*N:3*N][i] += f * 1.0
                    
                o3 = rs[i] - ys[i]
                if o3 > 0:
                    pen_val += o3 * o3
                    f = 2.0 * o3
                    dpen[N:2*N][i] += f * (-1.0/h)
                    dpen[2*N:3*N][i] += f * 1.0
                    
                o4 = rs[i] - (h - ys[i])
                if o4 > 0:
                    pen_val += o4 * o4
                    f = 2.0 * o4
                    dpen[N:2*N][i] += f * (1.0/h)
                    dpen[2*N:3*N][i] += f * 1.0
            
            val += 100.0 * pen_val
            grad = np.zeros_like(x)
            grad[2*N:3*N] = -1.0
            grad += 100.0 * dpen
            return val, grad
            
        try:
            res = minimize(obj_grad, x0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < 1.0: # Heuristic
                us, vs, rs = res.x[0:N], res.x[N:2*N], res.x[2*N:3*N]
                xs = us * w
                ys = vs * h
                # Check validity
                valid = True
                for i in range(N):
                    if rs[i] < 1e-6: valid = False
                    for j in range(i+1, N):
                        if rs[i] + rs[j] > np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) + 1e-6:
                            valid = False
                if valid:
                    score = np.sum(rs)
                    return score, np.column_stack([xs, ys, rs])
        except Exception:
            pass
        return -np.inf, None

    # Run loop
    w_values = np.linspace(0.5, 1.5, 11)
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    idx = 0
    for w in w_values:
        if time.time() > deadline: break
        for seed in seeds:
            if time.time() > deadline: break
            score, result = run_optimization_for_w(w, seed)
            if score > best_score:
                best_score = score
                best_result = result
            idx += 1
            # Limit total runs to fit in time
            if idx >= 30: break
        if idx >= 30: break

    if best_result is not None:
        return best_result
    else:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = [((i + 0.5) / side * w, (j + 0.5) / side * h, 0.5 / side * min(w, h) - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

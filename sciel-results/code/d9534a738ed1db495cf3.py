import numpy as np
import scipy.optimize as so
import time

N = 26

def construct():
    deadline = time.time() + 110.0
    best_score = -np.inf
    best_solution = None
    
    # Precompute pairs for penalty
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    
    def objective_and_grad(v):
        # v shape (3N,): [x0, y0, r0, x1, y1, r1, ...]
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Penalties
        pen = 0.0
        grad_pen = np.zeros_like(v)
        
        # Wall constraints
        # x >= r, y >= r, x+r <= 1, y+r <= 1
        # Violations:
        v1 = r - x      # x < r
        v2 = r - y      # y < r
        v3 = x + r - 1  # x > 1-r
        v4 = y + r - 1  # y > 1-r
        
        # Smooth penalty: 0.5 * max(0, v)^2
        # Gradient of 0.5 * max(0, v)^2 w.r.t v is max(0, v)
        
        # We use a large penalty weight
        w = 1000.0
        
        # Helper to add penalty
        def add_pen(viol, idx, sign_idx, sign_v, sign_r, sign_x, sign_y):
            pass # This is getting complicated with gradients manually.
            
        # Let's just compute the value and use finite difference gradient?
        # L-BFGS-B supports finite difference if jac is None.
        # But for 78 vars, finite diff is 78 evaluations per step. Might be slow.
        # Let's implement the gradient analytically for the penalty part.
        
        # Objective: -sum(r) + w * (Wall_Pen + Pair_Pen)
        
        # Gradients for -sum(r):
        grad_obj = np.zeros_like(v)
        grad_obj[2::3] = -1.0
        
        # Wall Penalties
        # v1 = r - x. d/dx = -1, d/dr = 1
        # v2 = r - y. d/dy = -1, d/dr = 1
        # v3 = x + r - 1. d/dx = 1, d/dr = 1
        # v4 = y + r - 1. d/dy = 1, d/dr = 1
        
        # Penalty P = 0.5 * sum(max(0, v)^2)
        # dP/dv = max(0, v)
        
        # Accumulate gradients
        # v1
        m1 = np.maximum(0, v1)
        grad_obj[0::3] -= m1  # d/dx
        grad_obj[2::3] += m1  # d/dr
        
        # v2
        m2 = np.maximum(0, v2)
        grad_obj[1::3] -= m2  # d/dy
        grad_obj[2::3] += m2  # d/dr
        
        # v3
        m3 = np.maximum(0, v3)
        grad_obj[0::3] += m3  # d/dx
        grad_obj[2::3] += m3  # d/dr
        
        # v4
        m4 = np.maximum(0, v4)
        grad_obj[1::3] += m4  # d/dy
        grad_obj[2::3] += m4  # d/dr
        
        pen_val = 0.5 * (np.sum(m1**2) + np.sum(m2**2) + np.sum(m3**2) + np.sum(m4**2))
        
        # Pair Penalties
        # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # viol_ij = r_i + r_j - d_ij
        # P_ij = 0.5 * max(0, viol_ij)^2
        
        # d/dxi = - (xi-xj)/d_ij * max(0, viol)
        # d/dxj = (xi-xj)/d_ij * max(0, viol)
        # d/dri = max(0, viol)
        
        # This loop is O(N^2). N=26, so 325 pairs.
        # Vectorization is possible but loops are fine for N=26.
        
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-12:
                d = 1e-12
            viol = r[i] + r[j] - d
            if viol > 0:
                m = viol
                # Add to penalty value
                pen_val += 0.5 * m * m
                
                # Add to gradients
                # d/dxi
                grad_obj[3*i]   -= m * dx / d
                # d/dxj
                grad_obj[3*j]   += m * dx / d
                # d/dyi
                grad_obj[3*i+1] -= m * dy / d
                # d/dyj
                grad_obj[3*j+1] += m * dy / d
                # d/dri
                grad_obj[3*i+2] += m
                # d/drj
                grad_obj[3*j+2] += m
        
        total_pen = w * pen_val
        total_val = -sum_r + total_pen
        total_grad = grad_obj + w * (grad_obj - grad_obj[2::3][None].T if False else 0) # Wait, I added grad_obj for obj part.
        
        # Correction: grad_obj currently holds gradient of -sum(r) + derivatives of penalties?
        # No, I initialized grad_obj with -1 for radii.
        # Then I added derivatives of penalties to grad_obj.
        # So grad_obj is the gradient of the whole function (without w multiplier on penalty part? No, I didn't multiply w in gradient accumulation).
        # I must multiply the penalty gradients by w.
        
        # Let's re-calculate cleanly.
        
        # Reset
        grad = np.zeros_like(v)
        # Objective -sum(r)
        grad[2::3] = -1.0
        
        # Penalties
        # Walls
        m1 = np.maximum(0, v1)
        grad[0::3] -= m1
        grad[2::3] += m1
        
        m2 = np.maximum(0, v2)
        grad[1::3] -= m2
        grad[2::3] += m2
        
        m3 = np.maximum(0, v3)
        grad[0::3] += m3
        grad[2::3] += m3
        
        m4 = np.maximum(0, v4)
        grad[1::3] += m4
        grad[2::3] += m4
        
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-12: d = 1e-12
            viol = r[i] + r[j] - d
            if viol > 0:
                m = viol
                grad[3*i]   -= m * dx / d
                grad[3*j]   += m * dx / d
                grad[3*i+1] -= m * dy / d
                grad[3*j+1] += m * dy / d
                grad[3*i+2] += m
                grad[3*j+2] += m
        
        # Multiply penalty part by w?
        # The gradients calculated above are for the penalty function P.
        # The total gradient is grad_obj_part + w * grad_penalty_part.
        # I mixed them.
        # Let's separate.
        
        grad_penalty = np.zeros_like(v)
        # Walls
        grad_penalty[0::3] -= m1
        grad_penalty[2::3] += m1
        grad_penalty[1::3] -= m2
        grad_penalty[2::3] += m2
        grad_penalty[0::3] += m3
        grad_penalty[2::3] += m3
        grad_penalty[1::3] += m4
        grad_penalty[2::3] += m4
        
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-12: d = 1e-12
            viol = r[i] + r[j] - d
            if viol > 0:
                m = viol
                grad_penalty[3*i]   -= m * dx / d
                grad_penalty[3*j]   += m * dx / d
                grad_penalty[3*i+1] -= m * dy / d
                grad_penalty[3*j+1] += m * dy / d
                grad_penalty[3*i+2] += m
                grad_penalty[3*j+2] += m
                
        total_grad = -np.ones_like(v) * 0 # Just zero
        total_grad[2::3] = -1.0
        total_grad += w * grad_penalty
        
        return -sum_r + w * pen_val, total_grad

    # Bounds
    lb = np.zeros(3*N)
    ub = np.zeros(3*N)
    # r > 0
    lb[2::3] = 1e-6
    ub[2::3] = 0.5
    # x, y in [0, 1]
    lb[0::3] = 0.0
    ub[0::3] = 1.0
    lb[1::3] = 0.0
    ub[1::3] = 1.0
    
    # Restarts
    num_restarts = 100
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialization
        # Grid 5x6 = 30, take 26
        # Or random
        # Let's use a jittered grid
        side_x = 5
        side_y = 6
        # Create grid points
        xs = (np.arange(side_x) + 0.5) / side_x
        ys = (np.arange(side_y) + 0.5) / side_y
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        
        # Select 26 random points? Or first 26?
        # First 26 might be correlated. Random selection is better for diversity.
        idx = np.random.choice(len(grid_x), N, replace=False)
        x0 = grid_x[idx]
        y0 = grid_y[idx]
        
        # Add jitter
        x0 += np.random.normal(0, 0.05, N)
        y0 += np.random.normal(0, 0.05, N)
        
        # Initial radii
        # Estimate max radius
        # r ~ 0.1
        r0 = np.full(N, 0.1)
        # Add noise
        r0 += np.random.normal(0, 0.02, N)
        r0 = np.clip(r0, 0.01, 0.4)
        
        v0 = np.concatenate([x0, y0, r0])
        # Ensure v0 is within bounds
        v0 = np.clip(v0, lb, ub)
        
        # Optimize
        res = so.minimize(objective_and_grad, v0, method='L-BFGS-B', jac=True, bounds=list(zip(lb, ub)), options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success:
            v_sol = res.x
            # Check validity
            x_sol = v_sol[0::3]
            y_sol = v_sol[1::3]
            r_sol = v_sol[2::3]
            
            valid = True
            # Check walls
            if np.any(x_sol < r_sol - 1e-5) or np.any(y_sol < r_sol - 1e-5):
                valid = False
            if np.any(x_sol + r_sol > 1 + 1e-5) or np.any(y_sol + r_sol > 1 + 1e-5):
                valid = False
            
            # Check pairs
            for i, j in pairs:
                d = np.sqrt((x_sol[i]-x_sol[j])**2 + (y_sol[i]-y_sol[j])**2)
                if d < r_sol[i] + r_sol[j] - 1e-5:
                    valid = False
                    break
            
            if valid:
                score = np.sum(r_sol)
                if score > best_score:
                    best_score = score
                    best_solution = np.column_stack((x_sol, y_sol, r_sol))
                    
    if best_solution is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
        
    return best_solution

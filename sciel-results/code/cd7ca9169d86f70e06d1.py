import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    best_value = 0.0
    best_result = None

    # Precompute pairs for efficiency
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def solve_packing(W, H, x0, max_iter=2000):
        """
        Solves packing in W x H rectangle using L-BFGS-B with penalty.
        Variables z = [x_1..x_N, y_1..y_N, r_1..r_N]
        Objective: -sum(r) + penalty * (sum of violations^2)
        """
        # We use a penalty method. 
        # To make it effective, we need to balance the objective and penalty.
        # A common trick is to start with a small penalty and increase it, 
        # or just use a large fixed penalty if the scaling is good.
        # Given L-BFGS-B is fast, we can afford a moderate penalty.
        
        # Better approach: Minimize F(z) = -sum(r) + lambda * sum(violations^2)
        # If lambda is too small, infeasible. If too large, slow convergence to feasible.
        # Let's try a moderate lambda and rely on the optimizer to find a trade-off.
        # Actually, a more robust method for packing is to minimize the sum of 
        # "soft" overlaps and boundary penetrations, while pushing radii up.
        
        # Let's define the function to minimize:
        # f(z) = -sum(r) + 100 * (sum(max(0, boundary_violation)^2) + sum(max(0, overlap)^2))
        
        # However, max(0, x)^2 is not differentiable at 0. 
        # L-BFGS-B requires gradients. 
        # We can use a smooth approximation or just use the square of the violation 
        # assuming the solver handles the kink reasonably well (it often does in practice 
        # if the kink is not at the optimum, or we use a soft-plus function).
        # Alternatively, we can use scipy.optimize.minimize with method='L-BFGS-B' 
        # and provide the gradient.
        
        # Let's implement the gradient manually for speed.
        
        def objective_and_grad(z):
            x = z[:N]
            y = z[N:2*N]
            r = z[2*N:]
            
            # Objective part: -sum(r)
            obj = -np.sum(r)
            grad_obj = np.zeros_like(z)
            grad_obj[2*N:] = -1.0
            
            # Penalty part
            # Boundary violations
            # Left: x < r  => viol = r - x
            # Right: x > W - r => viol = x - (W - r) = x - W + r
            # Bottom: y < r => viol = r - y
            # Top: y > H - r => viol = y - H + r
            
            # We will use soft violations: v = max(0, viol)
            # Gradient of max(0, v)^2 is 2*v if v>0 else 0
            
            pen = 0.0
            grad_pen = np.zeros_like(z)
            
            # Left boundary
            v_left = r - x
            mask = v_left > 0
            if np.any(mask):
                vl = v_left[mask]
                pen += np.sum(vl**2)
                # d(pen)/dx = -2*vl (for those where mask is true)
                grad_pen[:N][mask] -= 2.0 * vl
                # d(pen)/dr = 2*vl
                grad_pen[2*N:][mask] += 2.0 * vl
                
            # Right boundary
            v_right = x - (W - r)
            mask = v_right > 0
            if np.any(mask):
                vr = v_right[mask]
                pen += np.sum(vr**2)
                grad_pen[:N][mask] += 2.0 * vr
                grad_pen[2*N:][mask] -= 2.0 * vr
                
            # Bottom boundary
            v_bot = r - y
            mask = v_bot > 0
            if np.any(mask):
                vb = v_bot[mask]
                pen += np.sum(vb**2)
                grad_pen[N:2*N][mask] -= 2.0 * vb
                grad_pen[2*N:][mask] += 2.0 * vb
                
            # Top boundary
            v_top = y - (H - r)
            mask = v_top > 0
            if np.any(mask):
                vt = v_top[mask]
                pen += np.sum(vt**2)
                grad_pen[N:2*N][mask] += 2.0 * vt
                grad_pen[2*N:][mask] -= 2.0 * vt
                
            # Pairwise overlaps
            # dist(i,j) < r_i + r_j => viol = r_i + r_j - dist
            # dist = sqrt(dx^2 + dy^2)
            # viol^2 = (r_i + r_j - dist)^2
            # d(viol^2)/dx_i = 2*viol * (-(dx_ij)/dist) ... wait.
            # Let d = dist. v = r_i + r_j - d.
            # d(v^2)/dx_i = 2v * dv/dx_i.
            # dv/dx_i = - d(d)/dx_i = - (x_i - x_j)/d.
            # So d(v^2)/dx_i = 2v * (-(x_i - x_j)/d) = 2v * (x_j - x_i)/d.
            # d(v^2)/dr_i = 2v * 1 = 2v.
            
            for i, j in pairs:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                # Avoid division by zero
                if dist_sq < 1e-12:
                    dist = 1e-6
                else:
                    dist = np.sqrt(dist_sq)
                
                v = r[i] + r[j] - dist
                if v > 0:
                    pen += v**2
                    # Gradient for i
                    # dx term: 2*v * (x_j - x_i)/dist
                    # dy term: 2*v * (y_j - y_i)/dist
                    # dr term: 2*v
                    grad_factor = 2.0 * v / dist
                    grad_pen[i] += -grad_factor * (x[i] - x[j]) # Wait, x_j - x_i = -(x_i - x_j)
                    # Let's re-verify: 
                    # v = r_i + r_j - sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
                    # dv/dx_i = - (x_i - x_j) / dist
                    # d(v^2)/dx_i = 2v * dv/dx_i = 2v * (-(x_i - x_j)/dist)
                    # So grad_pen[i] += 2*v * (-(x_i - x_j)/dist)
                    grad_pen[i] += -2.0 * v * (dx / dist)
                    grad_pen[N+i] += -2.0 * v * (dy / dist)
                    
                    # Gradient for j
                    # dv/dx_j = - (x_j - x_i) / dist = (x_i - x_j) / dist
                    # d(v^2)/dx_j = 2v * (x_i - x_j) / dist
                    grad_pen[j] += 2.0 * v * (dx / dist)
                    grad_pen[N+j] += 2.0 * v * (dy / dist)
                    
                    # Radius gradients
                    grad_pen[2*N+i] += 2.0 * v
                    grad_pen[2*N+j] += 2.0 * v
            
            # Total objective and gradient
            # We need to scale the penalty. 
            # If radii are ~0.1, sum(r) ~ 2. 
            # Violations might be ~0.01, squared ~1e-4. 
            # We need the penalty to dominate when infeasible.
            # Let's use a dynamic penalty or a large fixed one.
            # A fixed penalty of 1000 usually works if violations are small.
            # But if we start with large violations, it might be huge.
            # Let's try lambda = 100.
            
            lambda_pen = 100.0
            total_obj = obj + lambda_pen * pen
            total_grad = grad_obj + lambda_pen * grad_pen
            
            return total_obj, total_grad

        # Bounds
        max_r = min(W, H) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0, W))
        for i in range(N):
            bounds.append((0, H))
        for i in range(N):
            bounds.append((0, max_r))
            
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                           jac=True, bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
            
            z = res.x
            x = z[:N]
            y = z[N:2*N]
            r = z[2*N:]
            
            # Check feasibility with tight tolerance
            feasible = True
            # Boundary
            if np.any(x < r - 1e-5) or np.any(x > W - r + 1e-5):
                feasible = False
            if np.any(y < r - 1e-5) or np.any(y > H - r + 1e-5):
                feasible = False
                
            # Pairwise
            if feasible:
                for i, j in pairs:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-5:
                        feasible = False
                        break
                        
            if feasible:
                val = np.sum(r)
                return val, z
            else:
                return 0.0, None
        except Exception:
            return 0.0, None

    # Strategy:
    # 1. Iterate over aspect ratios W.
    # 2. For each W, try multiple initial configurations (grid, jittered grid, random).
    # 3. Use L-BFGS-B which is fast.
    
    aspect_ratios = np.linspace(0.6, 1.4, 17) # More granular
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
            
        H = 2.0 - W
        if H <= 0:
            continue
            
        # Generate several initial guesses
        inits = []
        
        # 1. Grid
        cols = int(np.ceil(np.sqrt(N * W / H)))
        rows = int(np.ceil(N / cols))
        while cols * rows < N:
            cols += 1
            
        x_grid = np.zeros(N)
        y_grid = np.zeros(N)
        r_grid = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x_grid[i] = (c + 0.5) * (W / cols)
            y_grid[i] = (r_idx + 0.5) * (H / rows)
            
        min_dim = min(W / cols, H / rows)
        r_grid[:] = 0.4 * min_dim
        inits.append(np.concatenate([x_grid, y_grid, r_grid]))
        
        # 2. Jittered Grids
        for seed in range(5):
            if time.time() > deadline:
                break
            np.random.seed(seed)
            x_jit = x_grid + np.random.uniform(-0.05, 0.05, N)
            y_jit = y_grid + np.random.uniform(-0.05, 0.05, N)
            x_jit = np.clip(x_jit, 0, W)
            y_jit = np.clip(y_jit, 0, H)
            r_jit = r_grid.copy()
            inits.append(np.concatenate([x_jit, y_jit, r_jit]))
            
        # 3. Random
        for seed in range(3):
            if time.time() > deadline:
                break
            np.random.seed(seed + 100)
            x_rand = np.random.uniform(0, W, N)
            y_rand = np.random.uniform(0, H, N)
            # Small radii to start
            r_rand = np.full(N, 0.05)
            inits.append(np.concatenate([x_rand, y_rand, r_rand]))
            
        for x0 in inits:
            if time.time() > deadline:
                break
            
            # Run optimization
            val, z = solve_packing(W, H, x0, max_iter=500)
            
            if val > best_value:
                best_value = val
                x = z[:N]
                y = z[N:2*N]
                r = z[2*N:]
                best_result = np.column_stack((x, y, r))
                
    if best_result is None:
        # Fallback: Simple grid
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x[i] = (c + 0.5) / cols
            y[i] = (r_idx + 0.5) / rows
            r[i] = 0.09
        best_result = np.column_stack((x, y, r))
        best_value = np.sum(r)
        
    return best_result

import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_val = 1e10
    best_x = None
    
    def generate_grid_init(w_init, strategy=0):
        """Generate initial grid layout with different strategies"""
        h_init = 2.0 - w_init
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []
        
        for i in range(side):
            for j in range(side):
                if len(xs) >= N:
                    break
                if strategy == 0:
                    # Standard grid
                    cx = (i + 0.5) / side * w_init
                    cy = (j + 0.5) / side * h_init
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.9
                elif strategy == 1:
                    # Offset grid for better packing
                    offset_x = 0.5 / side * (i % 2) * w_init
                    offset_y = 0.5 / side * (j % 2) * h_init
                    cx = (i + 0.5) / side * w_init + offset_x * 0.3
                    cy = (j + 0.5) / side * h_init + offset_y * 0.3
                    cx = np.clip(cx, 0.1, w_init - 0.1)
                    cy = np.clip(cy, 0.1, h_init - 0.1)
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.85
                else:
                    # Randomized grid
                    cx = (i + np.random.uniform(0.2, 0.8)) / side * w_init
                    cy = (j + np.random.uniform(0.2, 0.8)) / side * h_init
                    cx = np.clip(cx, 0.05, w_init - 0.05)
                    cy = np.clip(cy, 0.05, h_init - 0.05)
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.8
                
                xs.append(cx)
                ys.append(cy)
                rs.append(r)
        
        while len(xs) < N:
            xs.append(w_init/2)
            ys.append(h_init/2)
            rs.append(0.05)
            
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])
    
    def objective(v):
        w = v[0]
        if w <= 0 or w >= 2:
            return 1e10
        h = 2.0 - w
        
        xs = v[1:N+1]
        ys = v[N+1:2*N+1]
        rs = v[2*N+1:3*N+1]
        
        # Penalty for negative radii
        if np.any(rs < 0):
            return 1e10
            
        # Boundary penalties
        pen = 0.0
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, rs - (w - xs))**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, rs - (h - ys))**2)
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if d < min_d:
                    pen += (min_d - d)**2
        
        return -np.sum(rs) + 100.0 * pen
    
    def gradient_flow_optimize(xs, ys, rs, w, h, num_steps=2000, dt=0.01):
        """Differential inclusion approach: gradient flow with projection"""
        xs = xs.copy()
        ys = ys.copy()
        rs = rs.copy()
        
        for step in range(num_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Compute gradients for maximizing sum of radii
            grad_xs = np.zeros(N)
            grad_ys = np.zeros(N)
            grad_rs = np.ones(N)  # d(sum rs)/d(rs_i) = 1
            
            # Overlap constraints: for each pair (i,j), if overlapping, push apart
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs[i] + rs[j]
                    
                    if d < min_d and d > 1e-10:
                        # Gradient of penalty: (min_d - d) * direction
                        overlap = min_d - d
                        # Push centers apart
                        grad_xs[i] += overlap * dx / d
                        grad_ys[i] += overlap * dy / d
                        grad_xs[j] -= overlap * dx / d
                        grad_ys[j] -= overlap * dy / d
                        # Reduce radii
                        grad_rs[i] -= overlap * 0.5
                        grad_rs[j] -= overlap * 0.5
            
            # Boundary constraints
            for i in range(N):
                # Left boundary
                if rs[i] > xs[i]:
                    grad_xs[i] -= (rs[i] - xs[i])
                    grad_rs[i] -= (rs[i] - xs[i]) * 0.5
                # Right boundary
                if rs[i] > w - xs[i]:
                    grad_xs[i] += (rs[i] - (w - xs[i]))
                    grad_rs[i] -= (rs[i] - (w - xs[i])) * 0.5
                # Bottom boundary
                if rs[i] > ys[i]:
                    grad_ys[i] -= (rs[i] - ys[i])
                    grad_rs[i] -= (rs[i] - ys[i]) * 0.5
                # Top boundary
                if rs[i] > h - ys[i]:
                    grad_ys[i] += (rs[i] - (h - ys[i]))
                    grad_rs[i] -= (rs[i] - (h - ys[i])) * 0.5
            
            # Semi-implicit Euler step
            xs += dt * grad_xs
            ys += dt * grad_ys
            rs += dt * grad_rs
            
            # Projection: ensure feasibility
            # 1. Keep centers in bounds
            for i in range(N):
                max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
                if max_r < 0:
                    # Move center to nearest valid position
                    xs[i] = np.clip(xs[i], 0, w)
                    ys[i] = np.clip(ys[i], 0, h)
                    max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = min(rs[i], max_r)
                rs[i] = max(rs[i], 0.0)
            
            # 2. Resolve overlaps by shrinking radii
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs[i] + rs[j]
                    if d < min_d - 1e-12:
                        if d > 1e-10:
                            factor = d / (min_d * 1.001)
                            rs[i] *= factor
                            rs[j] *= factor
                        else:
                            # Move apart
                            if dx == 0 and dy == 0:
                                xs[i] += 0.01
                                xs[j] -= 0.01
                            else:
                                # Push apart along existing direction
                                push = (min_d - d) / 2
                                xs[i] += push * dx / d
                                ys[i] += push * dy / d
                                xs[j] -= push * dx / d
                                ys[j] -= push * dy / d
            
            # 3. Final boundary check
            for i in range(N):
                max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = min(rs[i], max(0, max_r))
        
        return xs, ys, rs
    
    def polish_solution(x_sol):
        """Final high-precision polish pass"""
        w_best = x_sol[0]
        h_best = 2.0 - w_best
        xs_best = x_sol[1:N+1].copy()
        ys_best = x_sol[N+1:2*N+1].copy()
        rs_best = x_sol[2*N+1:3*N+1].copy()
        
        # Multiple passes to resolve collisions and ensure validity
        for _ in range(50):
            changed = False
            
            # Ensure within bounds
            for i in range(N):
                max_r = min(xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
                if rs_best[i] > max_r:
                    rs_best[i] = max_r
                    changed = True
            
            # Resolve overlaps iteratively
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs_best[i] - xs_best[j]
                    dy = ys_best[i] - ys_best[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs_best[i] + rs_best[j]
                    if d < min_d - 1e-12:
                        # Shrink both radii proportionally
                        if d > 1e-10:
                            factor = d / (min_d * 1.001)
                            rs_best[i] *= factor
                            rs_best[j] *= factor
                        else:
                            # Move apart if very close
                            if dx == 0 and dy == 0:
                                xs_best[i] += 0.01
                                xs_best[j] -= 0.01
                        changed = True
            
            if not changed:
                break
        
        # Final validation
        for i in range(N):
            rs_best[i] = min(rs_best[i], xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
            if rs_best[i] < 0:
                rs_best[i] = 0
        
        return np.column_stack((xs_best, ys_best, rs_best))
    
    # Try diverse aspect ratios with multiple initialization strategies
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    for w_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        for strategy in range(3):
            if time.time() > deadline:
                break
                
            xs, ys, rs = generate_grid_init(w_init, strategy)
            h_init = 2.0 - w_init
            
            # Use gradient flow optimization instead of L-BFGS-B
            try:
                xs_opt, ys_opt, rs_opt = gradient_flow_optimize(xs, ys, rs, w_init, h_init)
                
                # Compute objective value
                val = -np.sum(rs_opt)
                
                # Check validity
                valid = True
                for i in range(N):
                    if rs_opt[i] < 0 or rs_opt[i] > xs_opt[i] or rs_opt[i] > w_init - xs_opt[i] or rs_opt[i] > ys_opt[i] or rs_opt[i] > h_init - ys_opt[i]:
                        valid = False
                        break
                
                if valid and val < best_val:
                    best_val = val
                    best_x = np.concatenate(([w_init], xs_opt, ys_opt, rs_opt))
            except:
                pass
    
    if best_x is None:
        # Fallback to simple grid
        w_init = 1.0
        xs, ys, rs = generate_grid_init(w_init, 0)
        best_x = np.concatenate(([w_init], xs, ys, rs))
    
    # Final polish
    result = polish_solution(best_x)
    
    return result

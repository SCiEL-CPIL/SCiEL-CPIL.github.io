import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def _validate_solution(w, xs, ys, rs):
        """Check if solution is valid: no overlaps, within bounds, positive radii."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False
        if np.any(rs <= 0):
            return False
        # Check bounds
        if np.any(xs < rs) or np.any(xs > w - rs):
            return False
        if np.any(ys < rs) or np.any(ys > h - rs):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if d < rs[i] + rs[j] - 1e-9:
                    return False
        return True

    def _resolve_collisions(w, xs, ys, rs, max_iter=100):
        """Iteratively resolve collisions by shrinking radii."""
        h = 2.0 - w
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            resolved = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    min_d = rs[i] + rs[j]
                    if d < min_d:
                        resolved = False
                        if d > 1e-10:
                            factor = d / (min_d * 1.001)
                            rs[i] *= factor
                            rs[j] *= factor
                        else:
                            rs[i] *= 0.99
                            rs[j] *= 0.99
            # Ensure within bounds
            for i in range(N):
                rs[i] = min(rs[i], xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = max(rs[i], 0)
            if resolved:
                break
        return xs, ys, rs

    def _generate_hex_init(w, h):
        """Generate hexagonal lattice initial layout."""
        xs = []
        ys = []
        rs = []
        
        # Hexagonal packing with spacing
        spacing = 0.35
        row_h = spacing * np.sqrt(3) / 2
        max_rows = int(h / row_h) + 2
        max_cols = int(w / spacing) + 2
        
        for row in range(max_rows):
            y = row * row_h
            offset = (row % 2) * spacing / 2
            for col in range(max_cols):
                x = col * spacing + offset
                if x < w and y < h and len(xs) < N:
                    xs.append(x)
                    ys.append(y)
        
        while len(xs) < N:
            xs.append(w/2)
            ys.append(h/2)
        
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        
        # Compute radii based on local clearance
        rs = np.array([min(x, w-x, y, h-y) * 0.9 for x, y in zip(xs, ys)])
        
        return xs, ys, rs

    def _generate_grid_init(w, h):
        """Generate initial grid layout for given aspect ratio."""
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []
        for i in range(side):
            for j in range(side):
                if len(xs) >= N:
                    break
                cx = (i + 0.5) / side * w
                cy = (j + 0.5) / side * h
                r = min(cx, w - cx, cy, h - cy) * 0.85
                xs.append(cx)
                ys.append(cy)
                rs.append(r)
        while len(xs) < N:
            xs.append(w/2)
            ys.append(h/2)
            rs.append(0.05)
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def _slsqp_optimize(w_init, xs_init, ys_init, rs_init):
        """Optimize using SLSQP with explicit constraints."""
        h_init = 2.0 - w_init
        x0 = np.concatenate(([w_init], xs_init, ys_init, rs_init))
        
        def objective(v):
            w = v[0]
            h = 2.0 - w
            xs = v[1:N+1]
            ys = v[N+1:2*N+1]
            rs = v[2*N+1:3*N+1]
            return -np.sum(rs)
        
        # Constraints: non-overlap
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def make_overlap_constraint(i, j):
                    def constraint(v):
                        w = v[0]
                        h = 2.0 - w
                        xs = v[1:N+1]
                        ys = v[N+1:2*N+1]
                        rs = v[2*N+1:3*N+1]
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        return d - rs[i] - rs[j]
                    return constraint
                cons.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
        
        # Constraints: within bounds
        for i in range(N):
            def make_x_min(i):
                def constraint(v):
                    xs = v[1:N+1]
                    rs = v[2*N+1:3*N+1]
                    return xs[i] - rs[i]
                return constraint
            def make_x_max(i):
                def constraint(v):
                    w = v[0]
                    xs = v[1:N+1]
                    rs = v[2*N+1:3*N+1]
                    return w - xs[i] - rs[i]
                return constraint
            def make_y_min(i):
                def constraint(v):
                    ys = v[N+1:2*N+1]
                    rs = v[2*N+1:3*N+1]
                    return ys[i] - rs[i]
                return constraint
            def make_y_max(i):
                def constraint(v):
                    w = v[0]
                    h = 2.0 - w
                    ys = v[N+1:2*N+1]
                    rs = v[2*N+1:3*N+1]
                    return h - ys[i] - rs[i]
                return constraint
            def make_r_pos(i):
                def constraint(v):
                    rs = v[2*N+1:3*N+1]
                    return rs[i]
                return constraint
            
            cons.append({'type': 'ineq', 'fun': make_x_min(i)})
            cons.append({'type': 'ineq', 'fun': make_x_max(i)})
            cons.append({'type': 'ineq', 'fun': make_y_min(i)})
            cons.append({'type': 'ineq', 'fun': make_y_max(i)})
            cons.append({'type': 'ineq', 'fun': make_r_pos(i)})
        
        # Bounds
        w_min, w_max = 0.1, 1.9
        bounds = [(w_min, w_max)]
        for i in range(N):
            bounds.append((0.01, 1.99))  # x
        for i in range(N):
            bounds.append((0.01, 1.99))  # y
        for i in range(N):
            bounds.append((0.001, 1.0))  # r
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 500, 'ftol': 1e-12})
            if res.success or res.fun < -0.5:
                w_best = res.x[0]
                xs_best = res.x[1:N+1]
                ys_best = res.x[N+1:2*N+1]
                rs_best = res.x[2*N+1:3*N+1]
                
                # Polish the solution
                xs_best, ys_best, rs_best = _resolve_collisions(w_best, xs_best, ys_best, rs_best)
                
                if _validate_solution(w_best, xs_best, ys_best, rs_best):
                    return w_best, xs_best, ys_best, rs_best
        except Exception:
            pass
        
        return None

    def _refine_with_fixed_radii(w, xs_init, ys_init, rs_fixed):
        """
        Fix radii and optimize only positions to maximize minimum clearance.
        This is a convex problem: maximize t subject to:
          - distance between centers >= r_i + r_j + t
          - distance to walls >= r_i + t
        """
        h = 2.0 - w
        n_vars = 2 * N + 1  # xs, ys, t
        x0 = np.concatenate([xs_init, ys_init, [0.0]])
        
        def objective(v):
            return -v[-1]  # maximize t
        
        cons = []
        # Non-overlap constraints: d_ij >= r_i + r_j + t
        for i in range(N):
            for j in range(i+1, N):
                def make_overlap(i, j):
                    def constraint(v):
                        xs = v[:N]
                        ys = v[N:2*N]
                        t = v[2*N]
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        return d - rs_fixed[i] - rs_fixed[j] - t
                    return constraint
                cons.append({'type': 'ineq', 'fun': make_overlap(i, j)})
        
        # Wall constraints: distance to wall >= r_i + t
        for i in range(N):
            def make_wall_x_min(i):
                def constraint(v):
                    xs = v[:N]
                    t = v[2*N]
                    return xs[i] - rs_fixed[i] - t
                return constraint
            def make_wall_x_max(i):
                def constraint(v):
                    xs = v[:N]
                    t = v[2*N]
                    return w - xs[i] - rs_fixed[i] - t
                return constraint
            def make_wall_y_min(i):
                def constraint(v):
                    ys = v[N:2*N]
                    t = v[2*N]
                    return ys[i] - rs_fixed[i] - t
                return constraint
            def make_wall_y_max(i):
                def constraint(v):
                    ys = v[N:2*N]
                    t = v[2*N]
                    return h - ys[i] - rs_fixed[i] - t
                return constraint
            
            cons.append({'type': 'ineq', 'fun': make_wall_x_min(i)})
            cons.append({'type': 'ineq', 'fun': make_wall_x_max(i)})
            cons.append({'type': 'ineq', 'fun': make_wall_y_min(i)})
            cons.append({'type': 'ineq', 'fun': make_wall_y_max(i)})
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))
        for i in range(N):
            bounds.append((0.0, h))
        bounds.append((-0.1, 0.5))  # t
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 300, 'ftol': 1e-10})
            if res.success:
                xs_opt = res.x[:N]
                ys_opt = res.x[N:2*N]
                t_opt = res.x[2*N]
                
                # Check validity
                xs_chk, ys_chk = xs_opt.copy(), ys_opt.copy()
                if _validate_solution(w, xs_chk, ys_chk, rs_fixed):
                    return xs_opt, ys_opt
        except Exception:
            pass
        
        return None

    best_val = -1e10
    best_result = None
    
    # Try multiple aspect ratios with both hex and grid init
    for w_init in [1.0, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3, 0.65, 1.35]:
        if time.time() > deadline - 10:
            break
            
        h_init = 2.0 - w_init
        
        # Try hex init
        xs_hex, ys_hex, rs_hex = _generate_hex_init(w_init, h_init)
        result = _slsqp_optimize(w_init, xs_hex, ys_hex, rs_hex)
        if result is not None:
            w_best, xs_best, ys_best, rs_best = result
            val = np.sum(rs_best)
            if val > best_val:
                best_val = val
                best_result = (w_best, xs_best, ys_best, rs_best)
        
        # Try grid init
        xs_grid, ys_grid, rs_grid = _generate_grid_init(w_init, h_init)
        result = _slsqp_optimize(w_init, xs_grid, ys_grid, rs_grid)
        if result is not None:
            w_best, xs_best, ys_best, rs_best = result
            val = np.sum(rs_best)
            if val > best_val:
                best_val = val
                best_result = (w_best, xs_best, ys_best, rs_best)
    
    if best_result is None:
        # Fallback
        w = 1.0
        h = 1.0
        side = 5
        xs = np.array([(i+0.5)/side * w for i in range(side) for _ in range(side)][:N])
        ys = np.array([(j+0.5)/side * h for j in range(side) for _ in range(side)][:N])
        rs = np.array([min(x, w-x, y, h-y)*0.8 for x, y in zip(xs, ys)])
        best_result = (w, xs, ys, rs)
    
    w_best, xs_best, ys_best, rs_best = best_result
    
    # Final polish: try to improve by fixing radii and optimizing positions
    if time.time() < deadline - 5:
        refined = _refine_with_fixed_radii(w_best, xs_best, ys_best, rs_best)
        if refined is not None:
            xs_new, ys_new = refined
            if _validate_solution(w_best, xs_new, ys_new, rs_best):
                xs_best, ys_best = xs_new, ys_new
    
    # Final validation and cleanup
    xs_best, ys_best, rs_best = _resolve_collisions(w_best, xs_best, ys_best, rs_best)
    
    result = np.column_stack((xs_best, ys_best, rs_best))
    
    return result

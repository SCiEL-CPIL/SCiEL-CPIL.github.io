import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def _validate_solution(w, xs, ys, rs):
        """Check if solution is valid: no overlaps, within bounds, positive radii."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False
        if np.any(rs <= 0):
            return False
        # Check bounds
        if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9):
            return False
        if np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if d < rs[i] + rs[j] - 1e-9:
                    return False
        return True

    def _resolve_collisions(w, xs, ys, rs, max_iter=300):
        """Iteratively resolve collisions by adjusting positions and shrinking radii."""
        h = 2.0 - w
        xs = xs.copy()
        ys = ys.copy()
        rs = rs.copy()
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            resolved = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs[i] + rs[j]
                    if d < min_d - 1e-12:
                        resolved = False
                        if d > 1e-10:
                            overlap = (min_d - d) * 0.5
                            nx = dx / d
                            ny = dy / d
                            xs[i] += nx * overlap
                            ys[i] += ny * overlap
                            xs[j] -= nx * overlap
                            ys[j] -= ny * overlap
                        else:
                            xs[i] += 0.01 * np.random.randn()
                            ys[i] += 0.01 * np.random.randn()
                        # Shrink radii slightly to ensure separation
                        factor = min(1.0, d / (min_d * 1.001)) if d > 1e-10 else 0.998
                        rs[i] *= factor
                        rs[j] *= factor
            # Ensure within bounds
            for i in range(N):
                xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                rs[i] = min(rs[i], xs[i], w - xs[i], ys[i], h - ys[i])
                rs[i] = max(rs[i], 1e-6)
            if resolved:
                break
        return xs, ys, rs

    def _generate_hex_init(w, h):
        """Generate initial hexagonal cluster layout."""
        rows = [4, 5, 4, 5, 3]
        xs = []
        ys = []
        rs = []
        
        spacing = min(w, h) / 5.5
        offset = spacing * 0.866
        
        for row_idx, count in enumerate(rows):
            y = (row_idx + 0.5) * offset
            x_start = (w - (count - 1) * spacing) / 2
            for col in range(count):
                x = x_start + col * spacing
                r = min(x, w - x, y, h - y) * 0.85
                r = min(r, spacing * 0.45)
                xs.append(x)
                ys.append(y)
                rs.append(max(r, 0.05))
        
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def _generate_grid_init(w, h):
        """Generate initial grid layout."""
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
                r = min(cx, w - cx, cy, h - cy) * 0.8
                xs.append(cx)
                ys.append(cy)
                rs.append(max(r, 0.05))
        while len(xs) < N:
            xs.append(w/2)
            ys.append(h/2)
            rs.append(0.05)
        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def _generate_random_init(w, h):
        """Generate random initial layout."""
        xs = np.random.uniform(0.1, w-0.1, N)
        ys = np.random.uniform(0.1, h-0.1, N)
        rs = np.full(N, 0.1)
        return xs, ys, rs

    def _slsqp_optimize(w_init, xs_init, ys_init, rs_init, maxiter=500):
        """Optimize using SLSQP with explicit constraints."""
        h_init = 2.0 - w_init
        x0 = np.concatenate(([w_init], xs_init, ys_init, rs_init))
        
        def objective(v):
            rs = v[2*N+1:3*N+1]
            return -np.sum(rs)
        
        # Constraints: non-overlap
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def make_overlap_constraint(i, j):
                    def constraint(v):
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
                          options={'maxiter': maxiter, 'ftol': 1e-14})
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

    def _polish(w, xs, ys, rs, maxiter=300):
        """High-precision polish pass with tighter tolerance."""
        x0 = np.concatenate(([w], xs, ys, rs))
        
        def objective(v):
            rs = v[2*N+1:3*N+1]
            return -np.sum(rs)
        
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def make_overlap_constraint(i, j):
                    def constraint(v):
                        xs = v[1:N+1]
                        ys = v[N+1:2*N+1]
                        rs = v[2*N+1:3*N+1]
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        return d - rs[i] - rs[j]
                    return constraint
                cons.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
        
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
        
        bounds = [(0.1, 1.9)]
        for i in range(N):
            bounds.append((0.01, 1.99))
        for i in range(N):
            bounds.append((0.01, 1.99))
        for i in range(N):
            bounds.append((0.001, 1.0))
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': maxiter, 'ftol': 1e-16})
            if res.success or res.fun < objective(x0):
                w_best = res.x[0]
                xs_best = res.x[1:N+1]
                ys_best = res.x[N+1:2*N+1]
                rs_best = res.x[2*N+1:3*N+1]
                xs_best, ys_best, rs_best = _resolve_collisions(w_best, xs_best, ys_best, rs_best)
                if _validate_solution(w_best, xs_best, ys_best, rs_best):
                    return w_best, xs_best, ys_best, rs_best
        except Exception:
            pass
        return w, xs, ys, rs

    best_val = -1e10
    best_result = None
    
    # Try multiple aspect ratios with different initializations
    aspect_ratios = [1.0, 0.8, 1.25, 0.75, 1.33, 0.67, 1.5, 0.9, 1.1, 0.85, 1.15, 0.7, 1.4]
    
    for w_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        h_init = 2.0 - w_init
        
        # Try different initializations
        inits = [
            ('grid', _generate_grid_init(w_init, h_init)),
            ('hex', _generate_hex_init(w_init, h_init)),
            ('rand1', _generate_random_init(w_init, h_init)),
        ]
        
        for init_name, (xs_init, ys_init, rs_init) in inits:
            if time.time() > deadline:
                break
                
            result = _slsqp_optimize(w_init, xs_init, ys_init, rs_init, maxiter=400)
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
        rs = np.array([min(x, w-x, y, h-y)*0.7 for x, y in zip(xs, ys)])
        best_result = (w, xs, ys, rs)
    
    w_best, xs_best, ys_best, rs_best = best_result
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        w_best, xs_best, ys_best, rs_best = _polish(w_best, xs_best, ys_best, rs_best, maxiter=500)
    
    # Final collision resolution
    xs_best, ys_best, rs_best = _resolve_collisions(w_best, xs_best, ys_best, rs_best)
    
    # Construct output array (x, y, r)
    result = np.column_stack((xs_best, ys_best, rs_best))
    
    return result

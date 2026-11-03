"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _grid_init(w, h, nc, nr, stagger=False):
    """Generate initial circle positions in a grid pattern."""
    positions = []
    for j in range(nr):
        for i in range(nc):
            if len(positions) >= N:
                break
            if stagger and j % 2 == 1:
                x = (i + 0.5) * w / nc + 0.3 * w / nc
            else:
                x = (i + 0.5) * w / nc
            y = (j + 0.5) * h / nr
            positions.append([x, y])
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx - N / 2) * w / N * 0.5
        y = h / 2 + (idx - N / 2) * h / N * 0.5
        positions.append([x, y])
    return np.array(positions[:N])

def _local_optimize(w, h, x0, deadline):
    """Local optimization using L-BFGS-B with penalty method."""
    def objective(params):
        pos = params[:2 * N].reshape(N, 2)
        rad = params[2 * N:]
        
        # Compute penalties vectorized
        # Boundary penalties
        pen_x = np.maximum(0, rad - pos[:, 0]) + np.maximum(0, pos[:, 0] - (w - rad))
        pen_y = np.maximum(0, rad - pos[:, 1]) + np.maximum(0, pos[:, 1] - (h - rad))
        boundary_pen = np.sum(pen_x) + np.sum(pen_y)
        
        # Overlap penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)
        
        return -np.sum(rad) + 100.0 * (boundary_pen + overlap_pen)
    
    max_r = min(w, h) / 2
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x
        bounds.append((0, h))  # y
    for i in range(N):
        bounds.append((1e-4, max_r))  # radius
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6,
                                  'maxls': 50, 'maxcor': 10})
        return result
    except Exception:
        return None

def _optimize_with_variable_w(x0, deadline):
    """
    Optimize with variable width w (and h = 2 - w).
    x0 = [w, x1..xN, y1..yN, r1..rN]
    """
    def objective(v):
        w = v[0]
        if w <= 0.01 or w >= 1.99:
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

    bounds = [(0.01, 1.99)]  # w
    # x, y, r bounds are dynamic but we set loose bounds here to let the penalty handle it
    # Or we can update bounds. L-BFGS-B allows bounds.
    # However, since w changes, x in [0,w] changes.
    # It's safer to use penalty for bounds if we want to vary w.
    # Let's just use penalty for everything except w bounds.
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6,
                                  'maxls': 50, 'maxcor': 10})
        return result
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Strategy 1: Fixed aspect ratios, local optimization (from current program)
    aspect_ratios = [1.0, 1.5, 2.0/3, 0.75, 1.25, 1.333, 0.667, 0.5, 2.0, 1.1, 0.9]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        layouts = [
            (5, 4, True),
            (4, 5, True),
            (5, 4, False),
            (4, 5, False),
            (3, 7, False),
            (7, 3, False),
        ]
        
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 5.0:
                break
            
            positions = _grid_init(w, h, nc, nr, stagger)
            r0 = min(w, h) / max(nc, nr) * 0.35
            radii = np.full(N, r0)
            
            x0 = np.concatenate([positions.flatten(), radii])
            
            result = _local_optimize(w, h, x0, deadline)
            if result is not None and result.fun < 1e6:
                pos = result.x[:2 * N].reshape(N, 2)
                rad = result.x[2 * N:]
                
                circles = np.column_stack([pos, rad])
                if _validate(circles, w, h):
                    value = _total_radius(circles)
                    if value > best_value:
                        best_value = value
                        best_result = circles

    # Strategy 2: Variable aspect ratio optimization (from inspiration 1)
    # This allows the optimizer to find the optimal w and h simultaneously.
    if time.time() < deadline - 30.0:
        # Try a few initial aspect ratios for the variable w optimization
        for w_init in [1.0, 0.8, 1.2, 0.6, 1.4]:
            if time.time() > deadline - 20.0:
                break
            
            h_init = 2.0 - w_init
            # Grid init for this aspect
            side = 5 # 5x5 grid for 21 circles
            xs = []
            ys = []
            rs = []
            for i in range(side):
                for j in range(side):
                    if len(xs) >= N:
                        break
                    cx = (i + 0.5) / side * w_init
                    cy = (j + 0.5) / side * h_init
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.8
                    xs.append(cx)
                    ys.append(cy)
                    rs.append(r)
            while len(xs) < N:
                xs.append(w_init/2)
                ys.append(h_init/2)
                rs.append(0.1)
                
            xs = np.array(xs[:N])
            ys = np.array(ys[:N])
            rs = np.array(rs[:N])
            
            x0 = np.concatenate(([w_init], xs, ys, rs))
            
            result = _optimize_with_variable_w(x0, deadline)
            
            if result is not None and result.fun < 1e6:
                w_best = result.x[0]
                h_best = 2.0 - w_best
                xs_best = result.x[1:N+1]
                ys_best = result.x[N+1:2*N+1]
                rs_best = result.x[2*N+1:3*N+1]
                
                # Post-process to ensure validity
                # Shrink radii if overlaps or boundary violations
                valid = True
                for i in range(N):
                    # Boundary
                    rs_best[i] = min(rs_best[i], xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
                    if rs_best[i] < 1e-6:
                        rs_best[i] = 1e-6
                        valid = False # Flag but continue
                            
                # Overlap resolution (simple shrink)
                for _ in range(5):
                    changed = False
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = xs_best[i] - xs_best[j]
                            dy = ys_best[i] - ys_best[j]
                            d = np.sqrt(dx*dx + dy*dy)
                            min_d = rs_best[i] + rs_best[j]
                            if d < min_d:
                                # Shrink both
                                factor = d / (min_d * 1.01) if d > 0 else 0.99
                                rs_best[i] *= factor
                                rs_best[j] *= factor
                                changed = True
                    if not changed:
                        break
                
                circles = np.column_stack((xs_best, ys_best, rs_best))
                if _validate(circles, w_best, h_best):
                    value = _total_radius(circles)
                    if value > best_value:
                        best_value = value
                        best_result = circles

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

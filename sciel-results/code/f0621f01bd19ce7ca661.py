"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _softplus(x, eps=1e-4):
    """Numerically stable softplus approximation: max(0, x) smoothed."""
    return np.where(x > eps, x, eps * np.log1p(np.exp(x / eps)))

def _smooth_max(x, eps=1e-4):
    """Smooth max(0, x) using softplus."""
    return _softplus(x, eps)

def _check_validity(circles, w, h, tol=1e-7):
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

def _separate_circles(pos, rad, w, h, max_iter=50):
    """Separate overlapping circles and move them inside bounds."""
    pos = pos.copy()
    rad = rad.copy()
    
    for _ in range(max_iter):
        # Move circles inside bounds
        for i in range(N):
            rad[i] = min(rad[i], pos[i,0], w - pos[i,0], pos[i,1], h - pos[i,1])
            rad[i] = max(rad[i], 1e-6)
        
        # Resolve overlaps
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rad[i] + rad[j]
                if d < min_d and d > 1e-10:
                    # Push apart along the line connecting centers
                    overlap = min_d - d
                    nx, ny = dx/d, dy/d
                    push = overlap * 0.5
                    pos[i,0] += nx * push
                    pos[i,1] += ny * push
                    pos[j,0] -= nx * push
                    pos[j,1] -= ny * push
                    changed = True
                elif d <= 1e-10:
                    # Same position, push in random direction
                    angle = np.random.uniform(0, 2*np.pi)
                    push = min_d * 0.5
                    pos[i,0] += np.cos(angle) * push
                    pos[i,1] += np.sin(angle) * push
                    pos[j,0] -= np.cos(angle) * push
                    pos[j,1] -= np.sin(angle) * push
                    changed = True
        
        # Re-clamp to bounds after separation
        for i in range(N):
            pos[i,0] = np.clip(pos[i,0], rad[i], w - rad[i])
            pos[i,1] = np.clip(pos[i,1], rad[i], h - rad[i])
        
        if not changed:
            break
    
    return pos, rad

def _smooth_overlap_penalty(pos, rad, w, h, mu=1.0):
    """Smooth penalty for overlaps and boundary violations."""
    penalty = 0.0
    # Boundary penalties
    for i in range(N):
        penalty += max(0, rad[i] - pos[i,0])**2
        penalty += max(0, rad[i] - (w - pos[i,0]))**2
        penalty += max(0, rad[i] - pos[i,1])**2
        penalty += max(0, rad[i] - (h - pos[i,1]))**2
    
    # Overlap penalties (smooth)
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            d = np.sqrt(dx*dx + dy*dy)
            min_d = rad[i] + rad[j]
            if d < min_d:
                penalty += (min_d - d)**2
    
    return penalty

def _objective_and_grad(x, w, h, lin_points=None, mu=1.0):
    """
    Objective and gradient for the SC formulation.
    x = [x1..xN, y1..yN, r1..rN]
    lin_points: list of (i, j, dx, dy, d) for linearized constraints
    """
    xs = x[:N]
    ys = x[N:2*N]
    rs = x[2*N:3*N]
    
    obj = -np.sum(rs)
    grad = np.zeros(3*N)
    
    # Boundary constraints (linear)
    for i in range(N):
        # x >= r
        if xs[i] < rs[i]:
            obj += mu * (rs[i] - xs[i])**2
            grad[i] += -2 * mu * (rs[i] - xs[i])
            grad[2*N+i] += 2 * mu * (rs[i] - xs[i])
        # x + r <= w
        if xs[i] + rs[i] > w:
            obj += mu * (xs[i] + rs[i] - w)**2
            grad[i] += 2 * mu * (xs[i] + rs[i] - w)
            grad[2*N+i] += 2 * mu * (xs[i] + rs[i] - w)
        # y >= r
        if ys[i] < rs[i]:
            obj += mu * (rs[i] - ys[i])**2
            grad[N+i] += -2 * mu * (rs[i] - ys[i])
            grad[2*N+i] += 2 * mu * (rs[i] - ys[i])
        # y + r <= h
        if ys[i] + rs[i] > h:
            obj += mu * (ys[i] + rs[i] - h)**2
            grad[N+i] += 2 * mu * (ys[i] + rs[i] - h)
            grad[2*N+i] += 2 * mu * (ys[i] + rs[i] - h)
    
    # Linearized overlap constraints
    if lin_points is not None:
        for (i, j, dx_lin, dy_lin, d_lin) in lin_points:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            # Linearized constraint: dx*dx_lin + dy*dy_lin >= d_lin*(r_i + r_j)
            lhs = dx * dx_lin + dy * dy_lin
            rhs = d_lin * (rs[i] + rs[j])
            if lhs < rhs:
                violation = rhs - lhs
                obj += mu * violation**2
                # Gradient w.r.t. x_i, y_i, r_i, x_j, y_j, r_j
                grad[i] += -2 * mu * violation * dx_lin
                grad[N+i] += -2 * mu * violation * dy_lin
                grad[2*N+i] += -2 * mu * violation * d_lin
                grad[j] += 2 * mu * violation * dx_lin
                grad[N+j] += 2 * mu * violation * dy_lin
                grad[2*N+j] += -2 * mu * violation * d_lin
    
    return obj, grad

def _objective(x, w, h, lin_points=None, mu=1.0):
    """Objective only for scipy minimize."""
    obj, _ = _objective_and_grad(x, w, h, lin_points, mu)
    return obj

def _local_optimize(x0, w, h, lin_points=None, mu=1.0, maxiter=500):
    """Local optimization using L-BFGS-B with linearized constraints."""
    max_r = min(w, h) / 2
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x
        bounds.append((0, h))  # y
    for i in range(N):
        bounds.append((1e-4, max_r))  # radius
    
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                         args=(w, h, lin_points, mu),
                         options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8,
                                  'maxls': 50, 'maxcor': 10})
        return result
    except Exception:
        return None

def _polish(pos, rad, w, h, max_sweeps=20, tol=1e-8):
    """Polish the solution by resolving any remaining overlaps."""
    pos = pos.copy()
    rad = rad.copy()
    
    for sweep in range(max_sweeps):
        # Clamp to bounds
        for i in range(N):
            rad[i] = min(rad[i], pos[i,0], w - pos[i,0], pos[i,1], h - pos[i,1])
            rad[i] = max(rad[i], 1e-6)
        
        # Resolve overlaps with small adjustments
        max_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rad[i] + rad[j]
                if d < min_d:
                    max_overlap = max(max_overlap, min_d - d)
                    if d > 1e-10:
                        nx, ny = dx/d, dy/d
                        push = (min_d - d) * 0.5 * 1.01
                        pos[i,0] += nx * push
                        pos[i,1] += ny * push
                        pos[j,0] -= nx * push
                        pos[j,1] -= ny * push
        
        # Re-clamp
        for i in range(N):
            pos[i,0] = np.clip(pos[i,0], rad[i], w - rad[i])
            pos[i,1] = np.clip(pos[i,1], rad[i], h - rad[i])
            rad[i] = min(rad[i], pos[i,0], w - pos[i,0], pos[i,1], h - pos[i,1])
        
        if max_overlap < tol:
            break
    
    return pos, rad

def _grid_initialization(w, h, nc, nr, stagger=False):
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

def _random_initialization(w, h):
    """Random initial positions."""
    pos = np.random.uniform(0, 1, (N, 2))
    pos[:, 0] *= w
    pos[:, 1] *= h
    # Small radii initially
    max_r = min(w, h) / 2 * 0.3
    rad = np.random.uniform(0.01, max_r, N)
    return pos, rad

def _successive_convex_relaxation(pos, rad, w, h, max_sweeps=10, mu=1.0):
    """
    Successive convex relaxation: iteratively linearize non-overlap constraints
    and solve the resulting QP.
    """
    pos = pos.copy()
    rad = rad.copy()
    
    for sweep in range(max_sweeps):
        # Build linearization points from current configuration
        lin_points = []
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-10:
                    lin_points.append((i, j, dx, dy, d))
        
        # Build x vector
        x = np.concatenate([pos[:,0], pos[:,1], rad])
        
        # Solve
        result = _local_optimize(x, w, h, lin_points, mu, maxiter=300)
        
        if result is not None:
            x_new = result.x
            pos = np.column_stack([x_new[:N], x_new[N:2*N]])
            rad = x_new[2*N:3*N]
            
            # Separate and polish
            pos, rad = _separate_circles(pos, rad, w, h, max_iter=30)
            pos, rad = _polish(pos, rad, w, h, max_sweeps=10, tol=1e-8)
        
        # Check convergence
        if sweep > 0:
            change = np.max(np.abs(pos - x_new[:2*N].reshape(N,2))) + np.max(np.abs(rad - x_new[2*N:]))
            if change < 1e-8:
                break
    
    return pos, rad

def _best_known():
    """Return a known good configuration as fallback."""
    # 5x5 grid in a 1x1 square (perimeter 4)
    side = 5
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

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
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
    """Local optimization using L-BFGS-B with smooth penalty method."""
    def objective(params):
        pos = params[:2 * N].reshape(N, 2)
        rad = params[2 * N:]
        
        # Smooth boundary penalties using softplus
        pen_x = _smooth_max(rad - pos[:, 0]) + _smooth_max(pos[:, 0] - (w - rad))
        pen_y = _smooth_max(rad - pos[:, 1]) + _smooth_max(pos[:, 1] - (h - rad))
        boundary_pen = np.sum(pen_x) + np.sum(pen_y)
        
        # Smooth overlap penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                violation = min_dist - dist
                if violation > 0:
                    overlap_pen += _softplus(violation)
        
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
                         options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8,
                                  'maxls': 100, 'maxcor': 20})
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
            
        # Smooth boundary penalties
        pen = 0.0
        pen += np.sum(_softplus(rs - xs))
        pen += np.sum(_softplus(rs - (w - xs)))
        pen += np.sum(_softplus(rs - ys))
        pen += np.sum(_softplus(rs - (h - ys)))
        
        # Smooth overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                violation = min_d - d
                if violation > 0:
                    pen += _softplus(violation)
        
        return -np.sum(rs) + 100.0 * pen

    bounds = [(0.01, 1.99)]  # w
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8,
                                  'maxls': 100, 'maxcor': 20})
        return result
    except Exception:
        return None

def _hexagonal_init(w, h, nc, nr):
    """Generate hexagonal-like grid positions."""
    positions = []
    row_spacing = h / nr
    col_spacing = w / nc
    for j in range(nr):
        y = (j + 0.5) * row_spacing
        offset = 0.0 if j % 2 == 0 else 0.5 * col_spacing
        for i in range(nc):
            if len(positions) >= N:
                break
            x = (i + 0.5) * col_spacing + offset
            x = min(max(x, 0), w)
            positions.append([x, y])
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx - N / 2) * w / N * 0.3
        y = h / 2 + (idx - N / 2) * h / N * 0.3
        positions.append([x, y])
    return np.array(positions[:N])

def _high_precision_polish(circles, w, h, max_iter=100):
    """High-precision polish: iteratively resolve overlaps with tiny adjustments."""
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iter):
        # Clamp to bounds
        for i in range(N):
            rad[i] = min(rad[i], pos[i,0], w - pos[i,0], pos[i,1], h - pos[i,1])
            rad[i] = max(rad[i], 1e-8)
        
        max_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rad[i] + rad[j]
                if d < min_d:
                    overlap = min_d - d
                    max_overlap = max(max_overlap, overlap)
                    if d > 1e-12:
                        nx, ny = dx/d, dy/d
                        push = overlap * 0.51
                        pos[i,0] += nx * push
                        pos[i,1] += ny * push
                        pos[j,0] -= nx * push
                        pos[j,1] -= ny * push
        
        # Re-clamp
        for i in range(N):
            pos[i,0] = np.clip(pos[i,0], rad[i], w - rad[i])
            pos[i,1] = np.clip(pos[i,1], rad[i], h - rad[i])
            rad[i] = min(rad[i], pos[i,0], w - pos[i,0], pos[i,1], h - pos[i,1])
        
        if max_overlap < 1e-10:
            break
    
    return np.column_stack([pos, rad])

def _slsqp_refine(circles, w, h):
    """SLSQP-based refinement for final polish."""
    x0 = circles.flatten()
    
    def objective(x):
        pos = x[:2*N].reshape(N, 2)
        rad = x[2*N:]
        
        # Smooth penalties
        pen = 0.0
        pen += np.sum(_softplus(rad - pos[:, 0]))
        pen += np.sum(_softplus(rad - (w - pos[:, 0])))
        pen += np.sum(_softplus(rad - pos[:, 1]))
        pen += np.sum(_softplus(rad - (h - pos[:, 1])))
        
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rad[i] + rad[j]
                violation = min_d - d
                if violation > 0:
                    pen += _softplus(violation)
        
        return -np.sum(rad) + 50.0 * pen
    
    def constraints(x):
        pos = x[:2*N].reshape(N, 2)
        rad = x[2*N:]
        cons = []
        
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})  # x >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (w - x[i]) - x[2*N+i]})  # x + r <= w
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})  # y >= r
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (h - x[N+i]) - x[2*N+i]})  # y + r <= h
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                def cons_fn(x, i=i, j=j):
                    dx = x[i] - x[j]
                    dy = x[N+i] - x[N+j]
                    d = np.sqrt(dx*dx + dy*dy)
                    return d - (x[2*N+i] + x[2*N+j])
                cons.append({'type': 'ineq', 'fun': cons_fn})
        
        return cons
    
    max_r = min(w, h) / 2
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
    for i in range(N):
        bounds.append((1e-6, max_r))
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 50, 'ftol': 1e-12})
        if result.success and result.fun < 0:
            pos = result.x[:2*N].reshape(N, 2)
            rad = result.x[2*N:]
            circles_new = np.column_stack([pos, rad])
            if _validate(circles_new, w, h):
                return circles_new
    except Exception:
        pass
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None
    best_w = 1.0
    best_h = 1.0

    # Strategy 1: Fixed aspect ratios, local optimization
    aspect_ratios = [1.0, 1.5, 2.0/3, 0.75, 1.25, 1.333, 0.667, 0.5, 2.0, 1.1, 0.9, 1.4, 0.7]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15.0:
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
            (5, 5, True),
            (6, 4, True),
            (4, 6, True),
        ]
        
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 8.0:
                break
            
            if stagger:
                positions = _hexagonal_init(w, h, nc, nr)
            else:
                positions = _grid_init(w, h, nc, nr, stagger)
            r0 = min(w, h) / max(nc, nr) * 0.35
            radii = np.full(N, r0)
            
            x0 = np.concatenate([positions.flatten(), radii])
            
            result = _local_optimize(w, h, x0, deadline)
            if result is not None and result.fun < 1e6:
                pos = result.x[:2 * N].reshape(N, 2)
                rad = result.x[2 * N:]
                
                circles = np.column_stack([pos, rad])
                # Polish
                circles = _high_precision_polish(circles, w, h)
                
                if _validate(circles, w, h):
                    value = _total_radius(circles)
                    if value > best_value:
                        best_value = value
                        best_result = circles
                        best_w = w
                        best_h = h

    # Strategy 2: Variable aspect ratio optimization
    if time.time() < deadline - 25.0:
        for w_init in [1.0, 0.8, 1.2, 0.6, 1.4, 0.9, 1.1]:
            if time.time() > deadline - 18.0:
                break
            
            h_init = 2.0 - w_init
            # Use hexagonal init for better starting point
            positions = _hexagonal_init(w_init, h_init, 5, 4)
            radii = np.full(N, min(w_init, h_init) / 5 * 0.35)
            
            x0 = np.concatenate(([w_init], positions.flatten(), radii))
            
            result = _optimize_with_variable_w(x0, deadline)
            
            if result is not None and result.fun < 1e6:
                w_best = result.x[0]
                h_best = 2.0 - w_best
                xs_best = result.x[1:N+1]
                ys_best = result.x[N+1:2*N+1]
                rs_best = result.x[2*N+1:3*N+1]
                
                # Post-process to ensure validity
                for i in range(N):
                    rs_best[i] = min(rs_best[i], xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
                    if rs_best[i] < 1e-6:
                        rs_best[i] = 1e-6
                        
                # Overlap resolution
                for _ in range(10):
                    changed = False
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = xs_best[i] - xs_best[j]
                            dy = ys_best[i] - ys_best[j]
                            d = np.sqrt(dx*dx + dy*dy)
                            min_d = rs_best[i] + rs_best[j]
                            if d < min_d:
                                factor = d / (min_d * 1.005) if d > 0 else 0.995
                                rs_best[i] *= factor
                                rs_best[j] *= factor
                                changed = True
                    if not changed:
                        break
                
                circles = np.column_stack((xs_best, ys_best, rs_best))
                # High precision polish
                circles = _high_precision_polish(circles, w_best, h_best)
                
                if _validate(circles, w_best, h_best):
                    value = _total_radius(circles)
                    if value > best_value:
                        best_value = value
                        best_result = circles
                        best_w = w_best
                        best_h = h_best

    # Strategy 3: SLSQP refinement on best result
    if best_result is not None and time.time() < deadline - 5.0:
        refined = _slsqp_refine(best_result, best_w, best_h)
        if _validate(refined, best_w, best_h):
            value = _total_radius(refined)
            if value > best_value:
                best_value = value
                best_result = refined

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

import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """
    Check if the packing is valid:
    1. All circles inside the rectangle [0, W] x [0, H].
    2. No two circles overlap.
    """
    if W <= 0 or H <= 0:
        return False
    
    # Boundary check
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx * dx + dy * dy
            sum_r = r[i] + r[j]
            if dist_sq < sum_r * sum_r - tol:
                return False
    return True

def _generate_initial_positions(w, h, seed):
    """
    Generate initial positions using a relaxed grid.
    """
    rng = np.random.default_rng(seed)
    
    # Determine grid dimensions
    # Try to make cells roughly square
    aspect = w / h
    cols = int(np.round(np.sqrt(N * aspect)))
    rows = int(np.ceil(N / cols))
    
    # Ensure we have at least N cells
    while cols * rows < N:
        if w > h:
            cols += 1
        else:
            rows += 1
            
    # Create grid points
    x_grid = np.linspace(w / (2*cols), w - w / (2*cols), cols)
    y_grid = np.linspace(h / (2*rows), h - h / (2*rows), rows)
    
    xx, yy = np.meshgrid(x_grid, y_grid)
    xx = xx.flatten()[:N]
    yy = yy.flatten()[:N]
    
    # Add small random jitter to break symmetry
    xx += rng.uniform(-0.05, 0.05, N) * (w / cols)
    yy += rng.uniform(-0.05, 0.05, N) * (h / rows)
    
    # Clamp to bounds
    xx = np.clip(xx, 0.1, w - 0.1)
    yy = np.clip(yy, 0.1, h - 0.1)
    
    # Initial radii: small fraction of cell size
    cell_w = w / cols
    cell_h = h / rows
    r_init = 0.3 * min(cell_w, cell_h)
    r = np.full(N, r_init)
    
    # Simple relaxation to reduce initial overlaps
    # This helps SLSQP start from a less "violent" configuration
    for _ in range(20):
        for i in range(N):
            for j in range(i + 1, N):
                dx = xx[i] - xx[j]
                dy = yy[i] - yy[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if d < min_dist and d > 1e-6:
                    push = (min_dist - d) * 0.5
                    xx[i] += push * dx / d
                    yy[i] += push * dy / d
                    xx[j] -= push * dx / d
                    yy[j] -= push * dy / d
        xx = np.clip(xx, r, w - r)
        yy = np.clip(yy, r, h - r)
        
    return xx, yy, r

def _slsqp_optimize(x0, y0, r0, w, h, max_iter=300):
    """
    Optimize positions and radii using SLSQP.
    """
    z0 = np.concatenate([x0, y0, r0])
    
    def objective(z):
        r = z[2*N:]
        return -np.sum(r)
    
    def jac(z):
        g = np.zeros(3*N)
        g[2*N:] = -1.0
        return g
    
    # Constraints: g_c(z) >= 0
    
    def make_constraint(i, j):
        def fun(z):
            xi, yi, ri = z[i], z[N+i], z[2*N+i]
            xj, yj, rj = z[j], z[N+j], z[2*N+j]
            dx = xi - xj
            dy = yi - yj
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-10:
                return -100.0 # Strong penalty for coincidence
            return d - (ri + rj)
        return fun
    
    def make_wall_constraint(axis, side, i):
        def fun(z):
            if axis == 0:
                coord = z[i]
                dim = w
            else:
                coord = z[N+i]
                dim = h
            r = z[2*N+i]
            if side == 0:
                return coord - r
            else:
                return dim - coord - r
        return fun

    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(0, 0, i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(0, 1, i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(1, 0, i)})
        constraints.append({'type': 'ineq', 'fun': make_wall_constraint(1, 1, i)})
        
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, w - 0.01)) # x
    for i in range(N):
        bounds.append((0.01, h - 0.01)) # y
    for i in range(N):
        bounds.append((0.01, min(w, h)/2)) # r
        
    options = {'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
    
    try:
        res = minimize(objective, z0, method='SLSQP', jac=jac, 
                       bounds=bounds, constraints=constraints, options=options)
        if res.success or res.status == 1:
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            # Clamp to valid range just in case of numerical drift
            x_opt = np.clip(x_opt, r_opt, w - r_opt)
            y_opt = np.clip(y_opt, r_opt, h - r_opt)
            return x_opt, y_opt, r_opt
        else:
            # If it didn't converge, return the best feasible point found or initial
            # SLSQP often returns the last iterate even if not "successful"
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            x_opt = np.clip(x_opt, r_opt, w - r_opt)
            y_opt = np.clip(y_opt, r_opt, h - r_opt)
            return x_opt, y_opt, r_opt
    except Exception:
        return x0, y0, r0

def _polish_result(x, y, r, w, h, iterations=100, time_limit=10.0):
    """
    Local search: try to increase radii by moving circles slightly.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    start_time = time.time()
    
    for _ in range(iterations):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            # Try increasing radius slightly
            delta_r = 0.001
            r_trial = r.copy()
            r_trial[i] += delta_r
            
            # Check if valid
            valid = True
            # Check walls
            if x[i] - r_trial[i] < 0 or x[i] + r_trial[i] > w or y[i] - r_trial[i] < 0 or y[i] + r_trial[i] > h:
                valid = False
            
            # Check neighbors
            if valid:
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r_trial[i] + r[j]:
                        valid = False
                        break
            
            if valid:
                r[i] = r_trial[i]
                improved = True
            else:
                # Try moving circle i to make room
                # Simple 4-directional nudge
                for dx_move in [0.01, -0.01, 0.02, -0.02]:
                    for dy_move in [0.01, -0.01, 0.02, -0.02]:
                        x_trial = x.copy()
                        y_trial = y.copy()
                        x_trial[i] += dx_move
                        y_trial[i] += dy_move
                        
                        # Check if new position is valid with current radii
                        pos_valid = True
                        if x_trial[i] - r[i] < 0 or x_trial[i] + r[i] > w or y_trial[i] - r[i] < 0 or y_trial[i] + r[i] > h:
                            pos_valid = False
                        if pos_valid:
                            for j in range(N):
                                if i == j: continue
                                dx = x_trial[i] - x[j]
                                dy = y_trial[i] - y[j]
                                d = np.sqrt(dx*dx + dy*dy)
                                if d < r[i] + r[j]:
                                    pos_valid = False
                                    break
                                    
                        if pos_valid:
                            # Now try increasing radius at new position
                            r_trial2 = r.copy()
                            # Max radius at new pos
                            r_max = min(x_trial[i], w - x_trial[i], y_trial[i], h - y_trial[i])
                            for j in range(N):
                                if i == j: continue
                                dx = x_trial[i] - x[j]
                                dy = y_trial[i] - y[j]
                                d = np.sqrt(dx*dx + dy*dy)
                                r_max = min(r_max, d - r[j])
                            
                            if r_max > r[i] + 1e-4:
                                x[i] = x_trial[i]
                                y[i] = y_trial[i]
                                # Grow slowly to avoid jumping over local minima
                                r[i] = min(r_max, r[i] + 0.01) 
                                improved = True
                                break
                    if improved:
                        break
        if not improved:
            # Try larger steps if stuck? Or just break if no improvement for a while
            # For now, break if no improvement in this full pass
            break
            
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_score = 0.0
    best_params = None
    
    # Try different aspect ratios for the rectangle w x h
    # w + h = 2
    # Let ar = w/h. Then w = 2*ar/(1+ar), h = 2/(1+ar)
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 1.1, 1.3]
    
    for idx, ar in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        # Try a few random seeds for initialization
        for seed in range(3):
            if time.time() > deadline:
                break
                
            x0, y0, r0 = _generate_initial_positions(w, h, seed)
            
            # Run SLSQP
            x_opt, y_opt, r_opt = _slsqp_optimize(x0, y0, r0, w, h, max_iter=200)
            
            if time.time() > deadline:
                break
                
            # Verify validity
            if _is_valid(x_opt, y_opt, r_opt, w, h):
                # Polish
                x_pol, y_pol, r_pol = _polish_result(x_opt, y_opt, r_opt, w, h, iterations=50, time_limit=5.0)
                
                # Re-verify after polishing
                if _is_valid(x_pol, y_pol, r_pol, w, h):
                    score = np.sum(r_pol)
                    if score > best_score:
                        best_score = score
                        best_params = np.column_stack((x_pol, y_pol, r_pol))
                else:
                    # If polishing broke validity, use the SLSQP result
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_params = np.column_stack((x_opt, y_opt, r_opt))
                        
    if best_params is None:
        # Fallback to seed-like solution
        # Square packing in 1x1 box (perimeter 4)
        # 21 circles in a 5x5 grid (25 cells)
        # w=1, h=1
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    r = 0.5 / side - 1e-4
                    cells.append([cx, cy, r])
        return np.array(cells)
    else:
        return best_params

if __name__ == "__main__":
    result = construct()
    print(f"Sum of radii: {np.sum(result[:, 2])}")

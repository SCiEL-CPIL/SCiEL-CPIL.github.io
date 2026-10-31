"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _pack_grid(n_circles):
    """Pack n_circles in a unit square (perimeter 4) using a grid layout."""
    # Try to find a grid that fits n_circles
    best_val = 0
    best_arr = None
    for cols in range(1, n_circles + 1):
        rows = int(np.ceil(n_circles / cols))
        if cols * rows < n_circles:
            continue
        # We have cols x rows cells, use first n_circles
        # Radius limited by min spacing
        # In unit square with cols cols and rows rows:
        # cell width = 1/cols, cell height = 1/rows
        # radius = 0.5 * min(1/cols, 1/rows)
        r = 0.5 * min(1.0 / cols, 1.0 / rows)
        # But we can potentially do better with non-uniform
        # For now, uniform grid
        val = n_circles * r
        if val > best_val:
            best_val = val
            cells = []
            for i in range(cols):
                for j in range(rows):
                    if len(cells) >= n_circles:
                        break
                    cx = (i + 0.5) / cols
                    cy = (j + 0.5) / rows
                    cells.append([cx, cy, r])
                if len(cells) >= n_circles:
                    break
            best_arr = np.array(cells)
    if best_arr is None:
        # Fallback: single column
        r = 0.5 / n_circles
        best_arr = np.array([[0.5, (i + 0.5) / n_circles, r] for i in range(n_circles)])
    return best_arr


def _sum_radii(x, n):
    """Sum of radii from parameter vector x = [x1,y1,r1, x2,y2,r2, ...]."""
    r = x[2::3]
    return np.sum(r)


def _min_separation(x, n):
    """Minimum over all pairs of (dist - ri - rj)."""
    pos = x[:2 * n].reshape(n, 2)
    r = x[2::3]
    min_sep = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
            sep = d - r[i] - r[j]
            if sep < min_sep:
                min_sep = sep
    return min_sep


def _constraint_separations(x, n):
    """Return array of separation constraints (should be >= 0)."""
    pos = x[:2 * n].reshape(n, 2)
    r = x[2::3]
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
            cons.append(d - r[i] - r[j])
    return np.array(cons)


def _boundary_constraints(x, n):
    """Circles must be inside [0,1]x[0,1]."""
    pos = x[:2 * n].reshape(n, 2)
    r = x[2::3]
    cons = []
    for i in range(n):
        cons.append(pos[i, 0] - r[i])       # left
        cons.append(1.0 - pos[i, 0] - r[i]) # right
        cons.append(pos[i, 1] - r[i])       # bottom
        cons.append(1.0 - pos[i, 1] - r[i]) # top
    return np.array(cons)


def _local_search(x0, n, max_iter=500, step_scale=0.1):
    """Simple local search: perturb positions and radii, accept if valid and better."""
    x = x0.copy()
    best_val = _sum_radii(x, n)
    # Check initial validity
    if not np.all(_constraint_separations(x, n) >= -1e-9) or not np.all(_boundary_constraints(x, n) >= -1e-9):
        return x0, best_val
    
    step = step_scale
    for it in range(max_iter):
        improved = False
        for _ in range(10):
            x_trial = x.copy()
            # Perturb a random circle
            idx = np.random.randint(n)
            dx = np.random.uniform(-step, step)
            dy = np.random.uniform(-step, step)
            dr = np.random.uniform(-step * 0.5, step * 0.5)
            x_trial[3 * idx] += dx
            x_trial[3 * idx + 1] += dy
            x_trial[3 * idx + 2] = max(0.01, x_trial[3 * idx + 2] + dr)
            
            # Check constraints
            if np.all(_constraint_separations(x_trial, n) >= -1e-9) and np.all(_boundary_constraints(x_trial, n) >= -1e-9):
                val = _sum_radii(x_trial, n)
                if val > best_val + 1e-12:
                    x = x_trial
                    best_val = val
                    improved = True
        if not improved:
            step *= 0.9
            if step < 1e-6:
                break
    return x, best_val


def _slsqp_optimize(x0, n, maxiter=200):
    """Use SLSQP to maximize sum of radii subject to separation and boundary constraints."""
    def objective(x):
        return -_sum_radii(x, n)
    
    def constraints(x):
        return _constraint_separations(x, n)
    
    def boundary_cons(x):
        return _boundary_constraints(x, n)
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.5))  # r
    
    cons = [
        {'type': 'ineq', 'fun': constraints},
        {'type': 'ineq', 'fun': boundary_cons}
    ]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-10})
        if res.success:
            return res.x
        else:
            # Return best feasible point found
            if np.all(_constraint_separations(res.x, n) >= -1e-8) and np.all(_boundary_constraints(res.x, n) >= -1e-8):
                return res.x
            return x0
    except:
        return x0


def _expand_radii(x, n, max_iter=100):
    """Try to expand each circle's radius as much as possible while maintaining validity."""
    x = x.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            r_old = x[3 * i + 2]
            # Binary search for max radius
            lo = r_old
            hi = min(0.5, 1.0 - max(0.0, min(1.0, x[3*i])), 1.0 - max(0.0, min(1.0, x[3*i+1])))
            # Actually just try increasing
            for _try in range(20):
                mid = (lo + hi) / 2
                if mid <= r_old:
                    break
                x_trial = x.copy()
                x_trial[3 * i + 2] = mid
                if np.all(_constraint_separations(x_trial, n) >= -1e-10) and np.all(_boundary_constraints(x_trial, n) >= -1e-10):
                    lo = mid
                    x = x_trial
                    improved = True
                else:
                    hi = mid
        if not improved:
            break
    return x


def _force_relax(x, n, iterations=100, strength=0.1):
    """Force-directed relaxation to separate overlapping circles."""
    x = x.copy()
    for it in range(iterations):
        forces = np.zeros_like(x)
        pos = x[:2*n].reshape(n, 2)
        r = x[2::3].copy()
        
        for i in range(n):
            for j in range(i+1, n):
                d_vec = pos[i] - pos[j]
                d = np.sqrt(np.sum(d_vec**2))
                if d < 1e-10:
                    d_vec = np.array([0.01, 0.01])
                    d = np.sqrt(2) * 0.01
                min_dist = r[i] + r[j]
                if d < min_dist:
                    overlap = min_dist - d
                    force_dir = d_vec / d
                    forces[3*i] += strength * overlap * force_dir[0]
                    forces[3*i+1] += strength * overlap * force_dir[1]
                    forces[3*j] -= strength * overlap * force_dir[0]
                    forces[3*j+1] -= strength * overlap * force_dir[1]
        
        # Boundary forces
        for i in range(n):
            cx, cy, ri = x[3*i], x[3*i+1], x[3*i+2]
            if cx < ri:
                forces[3*i] += strength * (ri - cx)
            if cx > 1 - ri:
                forces[3*i] -= strength * (cx - (1-ri))
            if cy < ri:
                forces[3*i+1] += strength * (ri - cy)
            if cy > 1 - ri:
                forces[3*i+1] -= strength * (cy - (1-ri))
        
        x[:2*n] += forces[:2*n]
        # Clamp to boundary
        for i in range(n):
            ri = x[3*i+2]
            x[3*i] = np.clip(x[3*i], ri, 1-ri)
            x[3*i+1] = np.clip(x[3*i+1], ri, 1-ri)
    return x


def _is_valid(x, n, tol=1e-9):
    """Check if configuration is valid."""
    if not np.all(_constraint_separations(x, n) >= -tol):
        return False
    if not np.all(_boundary_constraints(x, n) >= -tol):
        return False
    if np.any(x[2::3] < 0):
        return False
    return True


def construct():
    start_time = time.time()
    
    # Start with a grid packing
    x0 = _pack_grid(N)
    # Convert to flat vector
    x = np.zeros(3 * N)
    for i in range(N):
        x[3*i] = x0[i, 0]
        x[3*i+1] = x0[i, 1]
        x[3*i+2] = x0[i, 2]
    
    best_x = x.copy()
    best_val = _sum_radii(x, N)
    
    # Force relax to fix any overlaps
    x = _force_relax(x, N, iterations=50, strength=0.5)
    if _is_valid(x, N):
        val = _sum_radii(x, N)
        if val > best_val:
            best_x = x.copy()
            best_val = val
    
    # SLSQP optimization
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        x_opt = _slsqp_optimize(x, N, maxiter=100)
        if _is_valid(x_opt, N, tol=1e-8):
            val = _sum_radii(x_opt, N)
            if val > best_val:
                best_x = x_opt.copy()
                best_val = val
    
    # Local search
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        x_ls, val_ls = _local_search(best_x, N, max_iter=300, step_scale=0.05)
        if val_ls > best_val:
            best_x = x_ls.copy()
            best_val = val_ls
    
    # Expand radii
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        x_exp = _expand_radii(best_x, N, max_iter=50)
        if _is_valid(x_exp, N, tol=1e-8):
            val = _sum_radii(x_exp, N)
            if val > best_val:
                best_x = x_exp.copy()
                best_val = val
    
    # Final SLSQP polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        x_final = _slsqp_optimize(best_x, N, maxiter=100)
        if _is_valid(x_final, N, tol=1e-8):
            val = _sum_radii(x_final, N)
            if val > best_val:
                best_x = x_final.copy()
                best_val = val
    
    # Final expand
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        x_final2 = _expand_radii(best_x, N, max_iter=30)
        if _is_valid(x_final2, N, tol=1e-8):
            val = _sum_radii(x_final2, N)
            if val > best_val:
                best_x = x_final2.copy()
                best_val = val
    
    # Convert back to (n, 3) array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_x[3*i]
        result[i, 1] = best_x[3*i+1]
        result[i, 2] = best_x[3*i+2]
    
    # Final validity check
    if not _is_valid(best_x, N, tol=1e-6):
        # Fallback to grid
        result = _pack_grid(N)
    
    return result

# EVOLVE-BLOCK-END

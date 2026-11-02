"""Circle packing in unit square for N=26 circles."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _get_delaunay_edges(points):
    """Get edges of Delaunay triangulation."""
    n = len(points)
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        if not edges:
            i_idx, j_idx = np.triu_indices(n, k=1)
            return i_idx, j_idx
        edge_list = sorted(edges)
        return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
    except Exception:
        i_idx, j_idx = np.triu_indices(n, k=1)
        return i_idx, j_idx


def _slsqp_optimize(circles, time_limit=30.0):
    """
    Optimize circle positions and radii using SLSQP with explicit inequality constraints.
    Variables: [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
    Objective: maximize sum(r) => minimize -sum(r)
    Constraints:
      - Boundary: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      - Non-overlap: dist(i,j) - r_i - r_j >= 0 for all pairs
    """
    n = N
    x0 = circles[:, 0].copy()
    y0 = circles[:, 1].copy()
    r0 = circles[:, 2].copy()
    
    x0 = np.clip(x0, 1e-6, 1 - 1e-6)
    y0 = np.clip(y0, 1e-6, 1 - 1e-6)
    r0 = np.clip(r0, 1e-6, 0.5)
    
    # Ensure initial feasibility by shrinking radii if needed
    for _ in range(20):
        candidate = np.column_stack([x0, y0, r0])
        if _validate_and_score(candidate) is not None:
            break
        r0 *= 0.95
    r0 = np.maximum(r0, 1e-6)
    
    start_params = np.concatenate([x0, y0, r0])
    
    # Get initial Delaunay edges
    points = np.column_stack([x0, y0])
    i_idx, j_idx = _get_delaunay_edges(points)
    
    # Also include all pairs as fallback for safety
    all_i, all_j = np.triu_indices(n, k=1)
    
    def objective(params):
        r = params[2*n:]
        return -np.sum(r)
    
    def gradient(params):
        grad = np.zeros(3*n)
        grad[2*n:] = -1.0
        return grad
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    def boundary_constraints(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        # r <= x
        c1 = x - r
        # r <= 1-x
        c2 = 1 - x - r
        # r <= y
        c3 = y - r
        # r <= 1-y
        c4 = 1 - y - r
        return np.concatenate([c1, c2, c3, c4])
    
    def boundary_jacobian(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        J = np.zeros((4*n, 3*n))
        # c1 = x - r: d/dx_i = 1, d/dr_i = -1
        J[:n, :n] = np.eye(n)
        J[:n, 2*n:] = -np.eye(n)
        # c2 = 1-x-r: d/dx_i = -1, d/dr_i = -1
        J[n:2*n, :n] = -np.eye(n)
        J[n:2*n, 2*n:] = -np.eye(n)
        # c3 = y - r: d/dy_i = 1, d/dr_i = -1
        J[2*n:3*n, n:2*n] = np.eye(n)
        J[2*n:3*n, 2*n:] = -np.eye(n)
        # c4 = 1-y-r: d/dy_i = -1, d/dr_i = -1
        J[3*n:, n:2*n] = -np.eye(n)
        J[3*n:, 2*n:] = -np.eye(n)
        return J
    
    # Non-overlap constraints: dist(i,j) - r_i - r_j >= 0
    def overlap_constraints(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - r[i_idx] - r[j_idx]
    
    def overlap_jacobian(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        m = len(i_idx)
        J = np.zeros((m, 3*n))
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        # d(dist)/dx_i = dx/dist, d(dist)/dx_j = -dx/dist
        J[:, i_idx] += dx / dist
        J[:, j_idx] -= dx / dist
        J[:, n + i_idx] += dy / dist
        J[:, n + j_idx] -= dy / dist
        # d(dist - r_i - r_j)/dr_i = -1, d/dr_j = -1
        J[:, 2*n + i_idx] -= 1.0
        J[:, 2*n + j_idx] -= 1.0
        return J
    
    # Combine all constraints
    def all_constraints(params):
        bc = boundary_constraints(params)
        oc = overlap_constraints(params)
        return np.concatenate([bc, oc])
    
    def all_jacobian(params):
        jb = boundary_jacobian(params)
        jo = overlap_jacobian(params)
        return np.vstack([jb, jo])
    
    bounds = [(1e-6, 1 - 1e-6)] * n + [(1e-6, 1 - 1e-6)] * n + [(1e-6, 0.5)] * n
    
    constraints = [{'type': 'ineq', 'fun': all_constraints, 'jac': all_jacobian}]
    
    start_time = time.time()
    
    try:
        res = minimize(
            objective, start_params, method='SLSQP',
            jac=gradient, bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        
        if time.time() - start_time > time_limit:
            return None, 0.0
        
        new_params = res.x
        new_x = np.clip(new_params[:n], 0, 1)
        new_y = np.clip(new_params[n:2*n], 0, 1)
        new_r = np.clip(new_params[2*n:], 1e-6, 0.5)
        
        # Clamp to boundary
        new_x = np.clip(new_x, new_r, 1 - new_r)
        new_y = np.clip(new_y, new_r, 1 - new_r)
        
        candidate = np.column_stack([new_x, new_y, new_r])
        score = _validate_and_score(candidate)
        
        if score is not None:
            return candidate, score
        else:
            return None, 0.0
            
    except Exception:
        return None, 0.0


def _polish_with_all_pairs(circles, time_limit=15.0):
    """Final polish using all pairs as constraints for maximum accuracy."""
    n = N
    x0 = circles[:, 0].copy()
    y0 = circles[:, 1].copy()
    r0 = circles[:, 2].copy()
    
    x0 = np.clip(x0, 1e-6, 1 - 1e-6)
    y0 = np.clip(y0, 1e-6, 1 - 1e-6)
    r0 = np.clip(r0, 1e-6, 0.5)
    
    # Ensure initial feasibility
    for _ in range(20):
        candidate = np.column_stack([x0, y0, r0])
        if _validate_and_score(candidate) is not None:
            break
        r0 *= 0.95
    r0 = np.maximum(r0, 1e-6)
    
    start_params = np.concatenate([x0, y0, r0])
    
    all_i, all_j = np.triu_indices(n, k=1)
    
    def objective(params):
        r = params[2*n:]
        return -np.sum(r)
    
    def gradient(params):
        grad = np.zeros(3*n)
        grad[2*n:] = -1.0
        return grad
    
    def boundary_constraints(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    
    def boundary_jacobian(params):
        J = np.zeros((4*n, 3*n))
        J[:n, :n] = np.eye(n)
        J[:n, 2*n:] = -np.eye(n)
        J[n:2*n, :n] = -np.eye(n)
        J[n:2*n, 2*n:] = -np.eye(n)
        J[2*n:3*n, n:2*n] = np.eye(n)
        J[2*n:3*n, 2*n:] = -np.eye(n)
        J[3*n:, n:2*n] = -np.eye(n)
        J[3*n:, 2*n:] = -np.eye(n)
        return J
    
    def overlap_constraints(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        dx = x[all_i] - x[all_j]
        dy = y[all_i] - y[all_j]
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - r[all_i] - r[all_j]
    
    def overlap_jacobian(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        m = len(all_i)
        J = np.zeros((m, 3*n))
        dx = x[all_i] - x[all_j]
        dy = y[all_i] - y[all_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        J[:, all_i] += dx / dist
        J[:, all_j] -= dx / dist
        J[:, n + all_i] += dy / dist
        J[:, n + all_j] -= dy / dist
        J[:, 2*n + all_i] -= 1.0
        J[:, 2*n + all_j] -= 1.0
        return J
    
    def all_constraints(params):
        bc = boundary_constraints(params)
        oc = overlap_constraints(params)
        return np.concatenate([bc, oc])
    
    def all_jacobian(params):
        jb = boundary_jacobian(params)
        jo = overlap_jacobian(params)
        return np.vstack([jb, jo])
    
    bounds = [(1e-6, 1 - 1e-6)] * n + [(1e-6, 1 - 1e-6)] * n + [(1e-6, 0.5)] * n
    constraints = [{'type': 'ineq', 'fun': all_constraints, 'jac': all_jacobian}]
    
    start_time = time.time()
    
    try:
        res = minimize(
            objective, start_params, method='SLSQP',
            jac=gradient, bounds=bounds, constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
        )
        
        if time.time() - start_time > time_limit:
            return None, 0.0
        
        new_params = res.x
        new_x = np.clip(new_params[:n], 0, 1)
        new_y = np.clip(new_params[n:2*n], 0, 1)
        new_r = np.clip(new_params[2*n:], 1e-6, 0.5)
        
        new_x = np.clip(new_x, new_r, 1 - new_r)
        new_y = np.clip(new_y, new_r, 1 - new_r)
        
        candidate = np.column_stack([new_x, new_y, new_r])
        score = _validate_and_score(candidate)
        
        if score is not None:
            return candidate, score
        else:
            return None, 0.0
            
    except Exception:
        return None, 0.0


def _initial_hexagonal(n=N):
    """Create a hexagonal grid initial configuration."""
    circles = np.zeros((n, 3))
    cols = 5
    rows = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.45
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    return circles


def _initial_grid(n=N):
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _initial_random(n=N, seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.01 + 0.02 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _make_valid(circles):
    """Ensure circles are valid by shrinking radii if needed."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    # Clamp positions
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Shrink radii until valid
    for _ in range(30):
        candidate = np.column_stack([x, y, r])
        if _validate_and_score(candidate) is not None:
            return candidate
        r *= 0.9
        # Also clamp to boundary
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 1e-6))
    
    # Last resort: very small radii
    r = np.full(N, 0.01)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    candidate = np.column_stack([x, y, r])
    if _validate_and_score(candidate) is not None:
        return candidate
    
    # Fallback: grid
    return _initial_grid(N)


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 140  # Safety margin before 150s kill
    
    best_circles = None
    best_score = 0.0
    
    # Initializations
    inits = [
        _initial_hexagonal(N),
        _initial_grid(N),
        _initial_random(N, seed=1),
        _initial_random(N, seed=2),
        _initial_random(N, seed=3),
        _initial_random(N, seed=4),
        _initial_random(N, seed=5),
    ]
    
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > time_budget - 25:
            break
        
        # Make valid
        valid_init = _make_valid(init)
        init_score = _validate_and_score(valid_init)
        if init_score is None:
            continue
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 20:
            continue
        
        # Phase 1: SLSQP with Delaunay constraints
        alloc1 = min(remaining * 0.4, 30)
        result1, score1 = _slsqp_optimize(valid_init, time_limit=alloc1)
        
        if result1 is not None and score1 > best_score:
            best_score = score1
            best_circles = result1.copy()
        
        # Phase 2: Polish with all pairs
        if best_circles is not None:
            remaining = time_budget - (time.time() - start_time)
            if remaining < 15:
                continue
            alloc2 = min(remaining * 0.5, 20)
            result2, score2 = _polish_with_all_pairs(best_circles, time_limit=alloc2)
            
            if result2 is not None and score2 > best_score:
                best_score = score2
                best_circles = result2.copy()
    
    # Final polish if we have time
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 10:
            result_final, score_final = _polish_with_all_pairs(best_circles, time_limit=min(remaining - 5, 15))
            if result_final is not None and score_final > best_score:
                best_score = score_final
                best_circles = result_final.copy()
    
    if best_circles is None:
        return _initial_grid(N)
    
    return best_circles

# EVOLVE-BLOCK-END
